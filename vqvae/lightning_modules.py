"""Pytorch lightning modules for training."""

import torch
import pytorch_lightning as pl


class VQVAEWrapper(pl.LightningModule):

    def __init__(self, model, learning_rate, learning_rate_codebook):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.learning_rate_codebook = learning_rate_codebook

    def training_step(self, batch, _):
        loss, reconstruction_loss, images, reconstructions = self.compute_loss(
            batch)
        self.log('codebook_loss', loss, prog_bar=True)
        self.log('reconstruction_loss', reconstruction_loss, prog_bar=True)
        return {
            'loss': loss + reconstruction_loss,
            'codebook_loss': loss,
            'reconstruction_loss': reconstruction_loss,
            'images': images,
            'reconstructions': reconstructions
        }

    def validation_step(self, batch, _):
        loss, reconstruction_loss, images, reconstructions = self.compute_loss(
            batch)
        self.log('codebook_loss_val', loss, on_epoch=True, prog_bar=True)
        self.log('reconstruction_loss_val',
                 reconstruction_loss,
                 on_epoch=True,
                 prog_bar=True)
        return {
            'codebook_loss_val': loss,
            'reconstruction_loss_val': reconstruction_loss,
            'images': images,
            'reconstructions': reconstructions
        }

    def compute_loss(self, batch):
        images = batch['image']
        reconstructions, _, _, loss = self.model(images)
        reconstruction_loss = torch.nn.MSELoss()(reconstructions, images)
        return loss, reconstruction_loss, images.detach(
        ), reconstructions.detach()

    def configure_optimizers(self):
        """Wrap the optimizer into a pl optimizer. """
        parameters = [{
            'params': self.model.encoder.parameters(),
            'lr': self.learning_rate
        }, {
            'params': self.model.decoder.parameters(),
            'lr': self.learning_rate
        }, {
            'params': self.model.quantizer.parameters(),
            'lr': self.learning_rate_codebook
        }]

        optimizer = torch.optim.Adam(parameters)
        return optimizer
