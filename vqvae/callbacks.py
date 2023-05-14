"""Callbacks used for training."""
import os

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

import warnings
import mlflow

import matplotlib.pyplot as plt


class ReconstructionVisualizer(Callback):
    """Plots reconstructions."""

    def __init__(self, run_id, log_frequency=50):
        self.run_id = run_id
        self.log_frequency = log_frequency
        self.epoch = 0
        self.validation_batches = 0

    def on_train_batch_end(self, trainer, _, output, __, ___):
        training_batches = trainer.global_step
        if training_batches % self.log_frequency == 0:
            self.plot_reconstructions(
                output,
                os.path.join('training_reconstructions',
                             f'batch_{training_batches:015}.png'))

    def on_train_epoch_end(self, _, __):
        self.epoch += 1
        self.validation_batches = 0

    def on_validation_batch_end(self, _, __, outputs, ___, ____):
        self.validation_batches += 1
        self.plot_reconstructions(
            outputs,
            os.path.join('validation_reconstructions', f'epoch_{self.epoch}',
                         f'batch_{self.validation_batches:015}.png'))

    def plot_reconstructions(self, output, path):
        num_plots = min(4, len(output['images']))
        original_images = [
            np.float32(output['images'][i].cpu().numpy())
            for i in range(num_plots)
        ]
        reconstructions = [
            np.float32(output['reconstructions'][i].cpu().numpy())
            for i in range(num_plots)
        ]
        fig, axs = plt.subplots(num_plots, 2, figsize=(6, 10))
        for i in range(num_plots):
            if num_plots > 1:
                axs[i, 0].imshow(original_images[i])
                axs[i, 1].imshow(reconstructions[i])
                axs[i, 0].axis('off')
                axs[i, 1].axis('off')
            else:
                axs[0].imshow(original_images[i])
                axs[1].imshow(reconstructions[i])
                axs[0].axis('off')
                axs[1].axis('off')
            plt.tight_layout()
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_figure(fig, path)
        plt.close(fig)


class LossMonitor(Callback):
    """Logs loss and model checkpoints to mlflow."""

    def __init__(self, save_dir, run_id, log_frequency=50):
        self.best_loss = float('inf')
        self.save_dir = save_dir
        self.run_id = run_id
        self.batch_rec_loss = []
        self.batch_codebook_loss = []
        self.log_frequency = log_frequency

    def on_train_epoch_end(self, trainer, _):
        # Getting the metrics.
        validation_loss = trainer.callback_metrics['reconstruction_loss_val']
        training_loss = trainer.callback_metrics['reconstruction_loss']
        validation_codebook_loss = trainer.callback_metrics[
            'codebook_loss_val']
        training_codebook_loss = trainer.callback_metrics['codebook_loss']
        metrics = {
            'metrics_validation_reconstruction_loss': validation_loss.item(),
            'metrics_training_reconstruction_loss': training_loss.item(),
            'metrics_validation_codebook_loss':
            validation_codebook_loss.item(),
            'metrics_training_codebook_loss': training_codebook_loss.item()
        }

        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_metrics(metrics, step=trainer.current_epoch)

            if validation_loss <= self.best_loss:
                self.best_loss = validation_loss

                path_encoder = os.path.join(self.save_dir, 'encoder.pt')
                path_quantizer = os.path.join(self.save_dir, 'quantizer.pt')
                path_decoder = os.path.join(self.save_dir, 'decoder.pt')
                torch.save([
                    trainer.model.model.encoder.state_dict(),
                    trainer.model.model.encoder.args
                ], path_encoder)
                torch.save([
                    trainer.model.model.quantizer.state_dict(),
                    trainer.model.model.quantizer.args
                ], path_quantizer)
                torch.save([
                    trainer.model.model.decoder.state_dict(),
                    trainer.model.model.decoder.args
                ], path_decoder)
                try:
                    mlflow.log_artifact(path_encoder,
                                        artifact_path='checkpoints')
                    mlflow.log_artifact(path_quantizer,
                                        artifact_path='checkpoints')
                    mlflow.log_artifact(path_decoder,
                                        artifact_path='checkpoints')
                # pylint: disable=broad-except
                except Exception as e:
                    warnings.warn(
                    'Something went wrong logging model to mlflow.\n'\
                    f'Failed with exception {e}')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx):
        # Getting the metrics.
        self.batch_rec_loss.append(outputs['reconstruction_loss'].detach())
        self.batch_codebook_loss.append(outputs['codebook_loss'].detach())

        # log to mlflow.
        rest = trainer.global_step % self.log_frequency
        if rest == 0 or trainer.is_last_batch:
            mean_rec_loss = torch.mean(torch.stack(self.batch_rec_loss))
            mean_codebook_loss = torch.mean(
                torch.stack(self.batch_codebook_loss))
            with mlflow.start_run(run_id=self.run_id):
                mlflow.log_metric(
                    key='metrics_training_batch_reconstruction_loss',
                    step=trainer.global_step,
                    value=mean_rec_loss)
                mlflow.log_metric(key='metrics_training_batch_codebook_loss',
                                  step=trainer.global_step,
                                  value=mean_codebook_loss)
            self.batch_rec_loss = []
            self.batch_codebook_loss = []
