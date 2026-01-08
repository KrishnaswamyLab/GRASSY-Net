import os, datetime
import numpy as np

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.utils.data

from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from models.GRASSY_model import GRASSY
from datasets.load_ZINC_tranche import ZINCDataset, Scattering

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--input_dim', default=None, type=int)
    parser.add_argument('--bottle_dim', default=25, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)

    parser.add_argument('--alpha', default=0.01, type=float)
    parser.add_argument('--beta', default=0.0005, type=float)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--len_epoch', default=None)

    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--save_dir', default='scripts/final_logs/', type=str)

    parser.add_argument('--GRASSY_version', default='AE+REG', type=str)
    parser.add_argument('--resume_from_checkpoint', default=None, type=str, help='Path to checkpoint file to resume from') # to alow rerun from a checkpoint

    # add args from trainer if available (older/newer PL versions differ)
    try:
        if hasattr(pl.Trainer, 'add_argparse_args'):
            parser = pl.Trainer.add_argparse_args(parser)
    except Exception:
        # ignore if not available
        pass

    # parse params
    args = parser.parse_args()

    if args.GRASSY_version == 'AE+REG':
        kl_div = False
        reg = True
    elif args.GRASSY_version == 'VAE+REG':
        kl_div = True
        reg = True
    elif args.GRASSY_version == 'AE':
        kl_div = False
        reg = False
    elif args.GRASSY_version == 'VAE':
        kl_div = True
        reg = False

    TRANCH = "ZINC12K"
    TRANCH_NAME = 'ZINC12K'
    full_dataset = ZINCDataset(f'datasets/{TRANCH}.npy', prop_stat_dict=f'datasets/{TRANCH}_stats.npy',
                                transform=Scattering(scatter_model_name=f'scripts/trained_models/{TRANCH_NAME}.npy'))
    if not kl_div:
        args.beta = 0
    if not reg:
        args.alpha = 0

    # Proper train/val/test split (80/10/10)
    # Original ZINC splits: train=10000, val=1000, test=1000
    train_size = 10000
    val_size = 1000
    test_size = len(full_dataset) - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # train loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                        shuffle=True, num_workers=15) 
    # valid loader 
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                        shuffle=False, num_workers=15)
    # test loader (for final evaluation)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                        shuffle=False, num_workers=15)

    # logger
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H-%M-%S")
    save_dir =  args.save_dir + TRANCH_NAME + f"{'_regress_' if reg else '_noregress_'}" + f"{'kld' if kl_div else 'nokld'}" + f"_{date_suffix}" +'/' # to keep all runs together

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Add wandb logger
    wandb_logger = WandbLogger(
        entity='grassy',
        project="GRASSY-ZINC12K",
        name=f"{TRANCH_NAME}_{'regress' if reg else 'noregress'}_{'kld' if kl_div else 'nokld'}",
        save_dir=save_dir,
    )

    # early stopping 
    early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=5,
            verbose=True,
            mode='min'
            )
    # Checkpoint callback - saves best validation model and last checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename='best-{epoch}-{val_loss:.3f}',
        monitor='val_loss',  # Monitor validation loss
        mode='min',  # Lower is better
        save_top_k=1,  # Save the best model
        save_last=True,  # Also save last checkpoint for resuming
    )
    # import pdb; pdb.set_trace()
    args.input_dim = len(train_set[0][0])
    args.len_epoch = len(train_loader)
    args.num_properties = len(train_set[0][1]) # <- allows varriyng number of properties
    print(args.input_dim)
    
    # init module
    model = GRASSY(hparams=args)
    # Log hyperparameters to wandb
    wandb_logger.log_hyperparams(vars(args)) # added this 

    # most basic trainer, uses good defaults. Use from_argparse_args if available, otherwise
    # construct Trainer with essential kwargs.
    try:
        if hasattr(pl.Trainer, 'from_argparse_args'):
            trainer = pl.Trainer.from_argparse_args(args,
                                        max_epochs=args.n_epochs,
                                        logger=wandb_logger, # added this 
                                         log_every_n_steps=1,# this too
                                        # gpus=args.n_gpus,
                                        callbacks=[checkpoint_callback] # commented out, early_stop_callback],
                                        )
        else:
            trainer = pl.Trainer(max_epochs=args.n_epochs,
                                logger=wandb_logger, # added this 
                                log_every_n_steps=1,# this too
                                #  gpus=args.n_gpus,
                                callbacks=[checkpoint_callback] # commented out, early_stop_callback],
                                 )
    except Exception:
        # fallback to direct construction
        trainer = pl.Trainer(max_epochs=args.n_epochs,
                            logger=wandb_logger, # added this 
                            log_every_n_steps=1,# this too
                            #  gpus=args.n_gpus,
                            callbacks=[checkpoint_callback] # commented out, early_stop_callback],
                             )

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader,
                ckpt_path=args.resume_from_checkpoint,
                )



    with torch.no_grad():
        loss = model.get_loss_list()

    #print('saving reconstruction loss')
    loss = np.array(loss)
    np.save(save_dir + f"{TRANCH_NAME}_{'noregress' if not reg else 'regress'}_{'nokld' if not kl_div  else 'kld'}_total_loss_list.npy", loss)
    # Save individual loss components
    recon_losses = np.array(model.get_recon_loss_list())
    reg_losses = np.array(model.get_reg_loss_list())
    kl_losses = np.array(model.get_kl_loss_list())

    np.save(save_dir + f"{TRANCH_NAME}_{'noregress' if not reg else 'regress'}_{'nokld' if not kl_div  else 'kld'}_recon_loss_list.npy", recon_losses)
    np.save(save_dir + f"{TRANCH_NAME}_{'noregress' if not reg else 'regress'}_{'nokld' if not kl_div  else 'kld'}_reg_loss_list.npy", reg_losses)
    np.save(save_dir + f"{TRANCH_NAME}_{'noregress' if not reg else 'regress'}_{'nokld' if not kl_div  else 'kld'}_kl_loss_list.npy", kl_losses)

    print('saving model')
    # Load best checkpoint (best validation performance)
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        # Load the best model
        best_model = GRASSY.load_from_checkpoint(best_model_path, hparams=args)
        # save it 
        model = best_model.cpu()
        model.dev_type = 'cpu'
        # Save the best model state dict
        torch.save(best_model.state_dict(), save_dir + f"{TRANCH_NAME}_{'noregress' if not reg else 'regress'}_{'nokld' if not kl_div  else 'kld'}_model.npy")
        print(f"Saved best model from epoch {checkpoint_callback.best_model_score}")
    else:
        # Fallback: save current model if no best checkpoint found
        model = model.cpu()
        model.dev_type = 'cpu'
        torch.save(model.state_dict(), save_dir + f"{TRANCH_NAME}_{'noregress' if not reg else 'regress'}_{'nokld' if not kl_div  else 'kld'}_model.npy")
    # Save model to Wandb
    wandb_logger.experiment.save(save_dir + f"{TRANCH_NAME}_{'noregress' if not reg else 'regress'}_{'nokld' if not kl_div  else 'kld'}_model.npy")


    no_transform_dataset = ZINCDataset(f'datasets/{TRANCH}.npy') # <- changed for consistency

    scat_mom_list = []
    prop = []
    qed = []
    heavywt = []
    tpsa = []
    ringcount = []
    # ki =  []
    
    prop.append(qed)
    prop.append(heavywt)
    prop.append(tpsa)
    prop.append(ringcount)
    # prop.append(ki)

    atom_percentage = []
    carbon = []
    nitro = []
    oxy = []
    atom_percentage.append(carbon)
    atom_percentage.append(nitro)
    atom_percentage.append(oxy)

    for index, entry in enumerate(tqdm(full_dataset)):

        scat_mom_list.append(entry[0].detach().cpu().numpy())
        qed.append(entry[1][0])
        heavywt.append(entry[1][1])
        tpsa.append(entry[1][6])
        ringcount.append(entry[1][9])
        # ki.append(entry[1][10])

        data = no_transform_dataset[index]

        c = 0
        n = 0
        o = 0
        i = 0
        for entry in data.element:
            if entry == 'C':
                c = c + 1
            if entry == 'N':
                n = n + 1
            if entry == 'O':
                o = o + 1
            i += 1
        
        c = c / i   
        n = n / i 
        o = o / i 
        carbon.append(c)
        nitro.append(n)
        oxy.append(o)
        
    scat_mom_list = np.array(scat_mom_list)

    moments = torch.Tensor(scat_mom_list)
    with torch.no_grad():
        ordered_embed = model.embed(moments)[0]

    print('saving embeddings')
    np.save(save_dir + f"ordered_embedding_{TRANCH_NAME}_{'noregress' if not reg else 'regress'}_{'nokld' if not kl_div  else 'kld'}.npy" , ordered_embed.cpu().detach().numpy() )
    np.save(save_dir + f"embedding_prop_lists_{TRANCH_NAME}_{'noregress' if not reg else 'regress'}_{'nokld' if not kl_div  else 'kld'}.npy", prop)
    np.save(save_dir + f"atom_percentages_{TRANCH_NAME}_{'noregress' if not reg else 'regress'}_{'nokld' if not kl_div  else 'kld'}.npy", atom_percentage)
