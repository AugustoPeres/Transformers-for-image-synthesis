"""Pytorch lightning modules for training."""

import torch
import pytorch_lightning as pl


class VQVAEWrapper(pl.LightningModule):

    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def training_step(self, batch, _):
        loss, reconstruction_loss = self.compute_loss(batch)
        self.log('codebook_loss', loss, prog_bar=True)
        self.log('reconstruction_loss', reconstruction_loss, prog_bar=True)
        return loss + reconstruction_loss

    def validation_step(self, batch, _):
        loss, reconstruction_loss = self.compute_loss(batch)
        self.log('codebook_loss_val', loss, on_epoch=True, prog_bar=True)
        self.log('reconstruction_loss_val',
                 reconstruction_loss,
                 on_epoch=True,
                 prog_bar=True)
        return {
            'codebook_loss_val': loss,
            'reconstruction_loss_val': reconstruction_loss
        }

    def compute_loss(self, batch):
        images = batch['image']
        reconstructions, _, _, loss = self.model(images)
        reconstruction_loss = torch.nn.MSELoss()(reconstructions, images)
        return loss, reconstruction_loss

    def configure_optimizers(self):
        """Wrap the optimizer into a pl optimizer. """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
