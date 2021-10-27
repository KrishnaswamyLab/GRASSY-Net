from models.GRASSY_model import GRASSY
from data.ZINCTranch import ZINCDataset, Scattering

from argparse import ArgumentParser
import datetime
import os
import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms



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
    parser.add_argument('--save_dir', default='final_logs/', type=str)

    parser.add_argument('--GRASSY_version', default='AE+REG', type=str)

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)
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

    
    TRANCH = "P14416_BindingDB_train"
    TRANCH_NAME = 'P14416'
    full_dataset = ZINCDataset(f'../final_tranches/{TRANCH}_subset.npy', prop_stat_dict=f'../final_tranches/{TRANCH}_subset_stats.npy',
                                transform=Scattering(scatter_model_name=f'../LEGS/final_models/{TRANCH_NAME}.npy'))

    if not kl_div:
        args.beta = 0
    if not reg:
        args.alpha = 0

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    #test_dataset = torch.utils.data.TensorDataset(*test_tup)

   
    # train loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                        shuffle=True, num_workers=15)
    # valid loader 
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                        shuffle=False, num_workers=15)

    # logger
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%M")
    save_dir =  args.save_dir + TRANCH_NAME + f"{'_regress_' if reg else '_noregress_'}" + f"{'kld' if kl_div else 'nokld'}" +'/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



    # early stopping 
    early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=5,
            verbose=True,
            mode='min'
            )

    args.input_dim = len(train_set[0][0])
    args.len_epoch = len(train_loader)
    print(args.input_dim)
    # init module
    model = GRASSY(hparams=args)

    # most basic trainer, uses good defaults
    trainer = pl.Trainer.from_argparse_args(args,
                                            max_epochs=args.n_epochs,
                                            gpus=args.n_gpus,
                                            callbacks=[early_stop_callback],
                                            #logger = logger
                                            )
    trainer.fit(model=model,
                train_dataloader=train_loader,
                val_dataloaders=valid_loader,
                )


    model = model.cpu()
    model.dev_type = 'cpu'

    with torch.no_grad():
        loss = model.get_loss_list()

    #print('saving reconstruction loss')
    loss = np.array(loss)
    np.save(save_dir + f"{TRANCH_NAME}_{'noregress' if not reg else 'regress'}_{'nokld' if not kl_div  else 'kld'}_reg_loss_list.npy", loss)


    print('saving model')
    torch.save(model.state_dict(), save_dir + f"{TRANCH_NAME}_{'noregress' if not reg else 'regress'}_{'nokld' if not kl_div  else 'kld'}_model.npy")



    no_transform_dataset = ZINCDataset(f'../final_tranches/{TRANCH}_subset.npy')

    scat_mom_list = []
    prop = []
    qed = []
    heavywt = []
    tpsa = []
    ringcount = []
    ki =  []
    prop.append(qed)
    prop.append(heavywt)
    prop.append(tpsa)
    prop.append(ringcount)
    prop.append(ki)


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
        ki.append(entry[1][10])

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


