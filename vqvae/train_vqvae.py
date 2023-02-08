"""Trains the vqvae"""

from absl import app
from absl import flags

import torch
import pytorch_lightning as pl

import modules
import datasets
import lightning_modules

FLAGS = flags.FLAGS

flags.DEFINE_string('path_to_data', None, 'Path to the data.')

flags.DEFINE_integer('in_channels', 3,
                     'Number of input channels. # color, 1 b&w')

flags.DEFINE_integer('channels', 256, 'The number of channels')

flags.DEFINE_integer('number_codebook_arrays', 512,
                     'The number of arrays in the codebook.')

flags.DEFINE_float('learning_rate', 1e-3, 'The learning rate of the model.')

flags.DEFINE_integer('batch_size', 128, 'Batch size.')

flags.DEFINE_boolean('use_gpu', False, 'Controls if the gpu is used.')

flags.DEFINE_integer('max_epochs', 10, 'The number of epochs to train for.')


def main(_):
    encoder = modules.Encoder(FLAGS.in_channels, FLAGS.channels)
    quantizer = modules.VectorQuantizer(FLAGS.number_codebook_arrays,
                                        FLAGS.channels, .25)
    decoder = modules.Decoder(FLAGS.channels, FLAGS.in_channels)
    model = modules.VQVAE(encoder, quantizer, decoder)

    dataset = datasets.ImageDataset(FLAGS.path_to_data)
    train_length = int(len(dataset) * .9)
    validation_length = len(dataset) - train_length
    training_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_length, validation_length],
        generator=torch.Generator().manual_seed(42))

    training_data_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        drop_last=True)
    validation_data_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        drop_last=False)

    lightning_module = lightning_modules.VQVAEWrapper(model,
                                                      FLAGS.learning_rate)

    accelerator = 'gpu' if FLAGS.use_gpu else None
    trainer = pl.Trainer(max_epochs=FLAGS.max_epochs, accelerator=accelerator)

    trainer.fit(lightning_module, training_data_loader, validation_data_loader)


if __name__ == '__main__':
    app.run(main)
