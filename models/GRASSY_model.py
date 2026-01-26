import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


class GRASSY(pl.LightningModule):

    def __init__(self, hparams):

        super(GRASSY, self).__init__()
        # Convert Namespace to dict if needed
        if hasattr(hparams, '__dict__') and not isinstance(hparams, dict):
            hparams = vars(hparams)
        
        self.save_hyperparameters(hparams)

        self.alpha = self.hparams.alpha
        self.beta = self.hparams.beta
        self.atom_loss_weight = getattr(self.hparams, 'atom_loss_weight', 1.0)
        
        self.input_dim = self.hparams.input_dim
        self.bottle_dim = self.hparams.bottle_dim
        self.hidden_dim = self.hparams.hidden_dim


        self.fc11 = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn11 = nn.BatchNorm1d(self.hidden_dim)
        
        self.fc12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn12 = nn.BatchNorm1d(self.hidden_dim)
        
        self.fc21 = nn.Linear(self.hidden_dim, self.bottle_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.bottle_dim)

        self.fc3 = nn.Linear(self.bottle_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)

        # property prediction (dynamic output size based on num_properties)
        self.num_properties = getattr(self.hparams, 'num_properties', 10)  # default to 10 for backward compatibility

        # Main regression head (all properties except num_atoms)
        self.regfc1 = nn.Linear(self.bottle_dim, 20)
        self.regbn1 = nn.BatchNorm1d(20)
        self.regfc2 = nn.Linear(20, self.num_properties - 1)  # All properties except num_atoms

        # Separate regression head for num_atoms
        self.atomregfc1 = nn.Linear(self.bottle_dim, 20)
        self.atomregbn1 = nn.BatchNorm1d(20)
        self.atomregfc2 = nn.Linear(20, 1)  # Predict num_atoms as regression

        self.loss_list = []
        self.recon_loss_list = []
        self.reg_loss_list = []
        self.kl_loss_list = []
                
        if self.hparams.n_gpus > 0:
            self.dev_type = 'cuda'

        if self.hparams.n_gpus == 0:
            self.dev_type = 'cpu'
        
        self.eps = 1e-5


    def kl_div(self,mu, logvar):

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return KLD
        
    def encode(self, x):

        h = self.fc11(x)
        h = self.bn11(h)
        h = F.gelu(h)
        h = self.fc12(h)
        h = self.bn12(h)
        h = F.gelu(h)
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):

        h3 = F.gelu(self.fc3(z))
        return self.fc4(h3)
    
    def embed(self, x):

        h = self.fc11(x)
        h = self.bn11(h)
        h = F.gelu(h)
        h = self.fc12(h)
        h = self.bn12(h)
        h = F.gelu(h)
        mu = self.fc21(h)
        logvar = self.fc22(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar 

    def predict(self, z):
        # Main regression head (all properties except num_atoms)
        h = self.regfc1(z)
        h = self.regbn1(h)
        h = F.gelu(h)
        y_pred = self.regfc2(h)
        # Separate regression head for num_atoms
        h_atom = self.atomregfc1(z)
        h_atom = self.atomregbn1(h_atom)
        h_atom = F.gelu(h_atom)
        num_atoms_pred = self.atomregfc2(h_atom)
        # Concatenate outputs: [regression_props..., num_atoms]
        y_full = torch.cat([y_pred, num_atoms_pred], dim=1)
        return y_full, y_pred, num_atoms_pred
    
    def predict_from_data(self,x):

        z = self.embed(x)[0]
        full_pred, _, _ = self.predict(z)
        return full_pred

    def forward(self, x):
        # encoding
        z, mu, logvar = self.embed(x)
        # predict
        y_full, y_pred, num_atoms_pred = self.predict(z)
        # recon
        x_hat = self.decode(z)
        return x_hat, y_full, mu, logvar, z

    def loss_multi_GRASSY(self, 
                        recon_x, x,  
                        mu, logvar,
                        y_full, y, 
                        alpha, beta, batch_idx):
        # reconstruction loss
        recon_loss = nn.MSELoss()(recon_x.flatten(), x.flatten()) 
        # regression loss for all properties except num_atoms
        reg_loss = nn.MSELoss()(y_full[:, :-1], y[:, :-1])
        # regression loss for num_atoms (last property)
        atom_reg_loss = nn.MSELoss()(y_full[:, -1], y[:, -1])
        # Weighted sum
        reg_loss = alpha * reg_loss.mean() + self.atom_loss_weight * atom_reg_loss.mean()
        # kl divergence 
        KLD = self.kl_div(mu, logvar)
        num_epochs = self.hparams.n_epochs - 5
        total_batches = self.hparams.len_epoch * num_epochs
        # loss annealing
        weight = min(1, float(self.trainer.global_step) / float(total_batches))
        kl_loss = weight * KLD
        kl_loss = beta * kl_loss
        total_loss = recon_loss + reg_loss + kl_loss
        self.loss_list.append(total_loss.item())
        self.recon_loss_list.append(recon_loss.item())
        self.reg_loss_list.append(reg_loss.item())
        self.kl_loss_list.append(kl_loss.item())
        log_losses = {'train_loss' : total_loss.detach(), 
                    'recon_loss' : recon_loss.detach(),
                    'pred_loss' :reg_loss.detach(),
                    'kl_loss': kl_loss.detach()
                    }
        return total_loss, log_losses

    def get_loss_list(self):
        return self.loss_list

    def get_recon_loss_list(self):
        return self.recon_loss_list

    def get_reg_loss_list(self):
        return self.reg_loss_list

    def get_kl_loss_list(self):
        return self.kl_loss_list

    def training_step(self, batch, batch_idx):
        x, y  = batch
        x = x.float()
        x_hat, y_full, mu, logvar, z = self(x)
        loss, log_losses = self.loss_multi_GRASSY(recon_x=x_hat, x=x, mu=mu, logvar=logvar, y_full=y_full, y=y,
                                                alpha=self.hparams.alpha, beta=self.hparams.beta, batch_idx=batch_idx)
        # Log metrics explicitly (required for newer PyTorch Lightning)
        self.log('train_loss', log_losses['train_loss'], on_step=True, on_epoch=True)
        self.log('recon_loss', log_losses['recon_loss'], on_step=True, on_epoch=True)
        self.log('pred_loss', log_losses['pred_loss'], on_step=True, on_epoch=True)
        self.log('kl_loss', log_losses['kl_loss'], on_step=True, on_epoch=True)
        return loss
   
    def validation_step(self, batch, batch_idx):
        x, y  = batch
        x = x.float()
        x_hat, y_full, mu, logvar, z = self(x)
        # reconstruction loss
        recon_loss = nn.MSELoss()(x_hat.flatten(), x.flatten())
        # regression loss for all properties except num_atoms
        reg_loss = nn.MSELoss()(y_full[:, :-1], y[:, :-1])
        # regression loss for num_atoms (last property)
        atom_reg_loss = nn.MSELoss()(y_full[:, -1], y[:, -1])
        # Weighted sum
        reg_loss = self.alpha * reg_loss.mean() + self.atom_loss_weight * atom_reg_loss.mean()
        # kl loss
        kl_loss = self.kl_div(mu, logvar)
        kl_loss = self.beta * kl_loss
        total_loss = recon_loss + reg_loss + kl_loss
        log_losses = {'val_loss' : total_loss.detach(), 
                    'val_recon_loss' : recon_loss.detach(),
                    'val_pred_loss' :reg_loss.detach(),
                    'val_kl_loss': kl_loss.detach()
                    }
        # Log metrics explicitly (required for checkpoint callback to monitor)
        self.log('val_loss', total_loss, on_step=False, on_epoch=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('val_pred_loss', reg_loss, on_step=False, on_epoch=True)
        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True)
        return log_losses

    # def on_validation_epoch_end(self, outputs):

    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_reconloss = torch.stack([x['val_recon_loss'] for x in outputs]).mean()
    #     avg_regloss = torch.stack([x['val_pred_loss'] for x in outputs]).mean()
    #     avg_klloss = torch.stack([x['val_kl_loss'] for x in outputs]).mean()

    #     tensorboard_logs = {'val_loss': avg_loss,
    #                         'val_avg_recon_loss': avg_reconloss,
    #                         'val_avg_pred_loss':avg_regloss,
    #                         'val_avg_kl_loss':avg_klloss
    #                         }

    #     self.log('val_loss', avg_loss.detach())
    #     print(avg_loss.detach())

    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
