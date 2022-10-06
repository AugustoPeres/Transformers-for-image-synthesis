"""Callbacks"""
import os

import mlflow

from pytorch_lightning.callbacks import Callback


class MLFlow(Callback):
    """Logs metrics and modules to mlflow."""

    def __init__(self, save_dir):
        self.best_loss = float('inf')
        self.save_dir = save_dir

    def on_train_epoch_end(self, trainer, _):
        # Getting the metrics.
        validation_loss = trainer.callback_metrics['val_loss']
        training_loss = trainer.callback_metrics['loss']

        mlflow.log_metric('training loss',
                          training_loss,
                          step=trainer.current_epoch)

        mlflow.log_metric('validation loss',
                          validation_loss,
                          step=trainer.current_epoch)

        if validation_loss <= self.best_loss:
            self.best_loss = validation_loss
            path = os.path.join(self.save_dir, 'checkpoint.ckpt')
            trainer.save_checkpoint(path)
            mlflow.log_artifact(path, artifact_path='checkpoints')
