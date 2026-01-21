
import torch
import torch.nn as nn
import pytorch_lightning as pl

class EndToEndScatteringGRASSYWrapper(pl.LightningModule):
    """
    Wrapper that combines learnable scattering transform with GRASSY.
    
    Trains the entire pipeline end-to-end:
    - Learnable scattering transform with MLPs
    - GRASSY autoencoder
    """
    
    def __init__(self, scattering_transform, grassy_model, hparams, alpha, beta):
        super().__init__()
        self.scattering = scattering_transform
        self.grassy = grassy_model
        self.hparams_config = hparams
        self.alpha = alpha
        self.beta = beta
        
        # Store loss histories
        self.total_loss_list = []
        self.recon_loss_list = []
        self.reg_loss_list = []
        self.kl_loss_list = []
    
    def forward(self, x):
        """Forward pass: scattering -> GRASSY forward."""
        # Compute scattering coefficients
        scattering_coeffs = self.scattering(x)
        # Pass through GRASSY
        return self.grassy(scattering_coeffs)
    
    def training_step(self, batch, batch_idx):
        """Training step: compute losses and backprop."""
        # batch is a Batch object from torch_geometric
        data = batch
        
        # Extract properties from PyG Data object
        if hasattr(data, 'y') and data.y is not None:
            y = data.y
        else:
            y = torch.zeros(data.num_graphs, dtype=torch.float32, device=data.x.device)
        
        # Compute scattering coefficients
        scattering_coeffs = self.scattering(data)
        # Forward pass through GRASSY
        x_hat, y_hat, mu, logvar, z = self.grassy(scattering_coeffs)
        
        # Compute individual losses (same as GRASSY model)
        recon_loss = nn.MSELoss()(x_hat.flatten(), scattering_coeffs.flatten())
        reg_loss = nn.MSELoss()(y_hat, y)
        kl_loss = self.grassy.kl_div(mu, logvar)
        
        # Apply weighting
        num_epochs = self.hparams_config.n_epochs - 5
        total_batches = self.hparams_config.len_epoch * num_epochs
        weight = min(1, float(self.trainer.global_step) / float(total_batches))
        
        kl_loss_weighted = self.beta * weight * kl_loss
        reg_loss_weighted = self.alpha * reg_loss.mean()
        
        total_loss = recon_loss + reg_loss_weighted + kl_loss_weighted
        
        # Store losses
        self.total_loss_list.append(total_loss.detach().item())
        self.recon_loss_list.append(recon_loss.detach().item())
        self.reg_loss_list.append(reg_loss_weighted.detach().item())
        self.kl_loss_list.append(kl_loss_weighted.detach().item())
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_recon_loss', recon_loss)
        self.log('train_reg_loss', reg_loss_weighted)
        self.log('train_kl_loss', kl_loss_weighted)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step: compute validation losses."""
        # batch is a Batch object from torch_geometric
        data = batch
        
        # Extract properties from PyG Data object
        if hasattr(data, 'y') and data.y is not None:
            y = data.y
        else:
            y = torch.zeros(data.num_graphs, dtype=torch.float32, device=data.x.device)
        
        # Compute scattering coefficients
        scattering_coeffs = self.scattering(data)
        
        # Forward pass through GRASSY
        x_hat, y_hat, mu, logvar, z = self.grassy(scattering_coeffs)
        
        # Compute individual losses
        recon_loss = nn.MSELoss()(x_hat.flatten(), scattering_coeffs.flatten())
        reg_loss = nn.MSELoss()(y_hat.reshape(-1), y.reshape(-1))
        kl_loss = self.grassy.kl_div(mu, logvar)
        
        # Apply weighting
        reg_loss_weighted = self.alpha * reg_loss.mean()
        kl_loss_weighted = self.beta * kl_loss
        
        total_loss = recon_loss + reg_loss_weighted + kl_loss_weighted
        
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_recon_loss', recon_loss)
        self.log('val_reg_loss', reg_loss_weighted)
        self.log('val_kl_loss', kl_loss_weighted)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer for all learnable parameters."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams_config.learning_rate
        )
        return optimizer
    
    def get_loss_list(self):
        return self.total_loss_list
    
    def get_recon_loss_list(self):
        return self.recon_loss_list
    
    def get_reg_loss_list(self):
        return self.reg_loss_list
    
    def get_kl_loss_list(self):
        return self.kl_loss_list
