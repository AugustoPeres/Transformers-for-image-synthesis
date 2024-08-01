import os
import pickle
import random
import tempfile

from absl import app
from absl import flags
from absl import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import dataset
from transformer import SeqTransformer
from callbacks import MLFlow

import mlflow

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 128, 'Size of the mini batches.')

flags.DEFINE_integer('max_epochs', 200, 'Number of epochs to train.')

flags.DEFINE_integer('dim_model', 64, 'Dimension of the transformer.')

flags.DEFINE_integer('num_attention_heads', 4,
                     'Number of attention heads in each layer.')

flags.DEFINE_integer('num_encoder_layers', 2, 'Number of encoder layers.')

flags.DEFINE_integer(
    'dim_feedforward', 64,
    'Dimension of the feed-forward network in the encoder layers.')

flags.DEFINE_float('learning_rate', 1e-3,
                   'The learning rate of the optimizer.')

flags.DEFINE_integer('window_size', 8, 'The size of the attention window.')
flags.DEFINE_integer(
    'latent_dim', 32,
    'Dimension of the latent space. That is, the side square dimension.')

flags.DEFINE_float('dropout', .0, 'Dropout')

flags.DEFINE_integer(
    'max_sequence_len', 200,
    'Maximum length of the target sequence when using the model for inference')

flags.DEFINE_string('path_to_data', None,
                    'Path to the file containing the training data.')

flags.DEFINE_float('train_validation_ratio', 0.9,
                   'Ratio of the dataset to use for validation.')

flags.DEFINE_integer('num_cpus_per_worker', 1,
                     'The number of cpus for each worker.')

flags.DEFINE_boolean('use_gpu', False, 'Controls if the gpu is used.')

flags.DEFINE_integer(
    'early_stopping_patience', 10,
    'How many epochs to wait for improvement before stopping.')

flags.mark_flag_as_required('path_to_data')


def _make_dataset_and_loader(data, source_vocabulary, batch_size, shuffle):
    data_set = dataset.SeqDataSet(data, source_vocabulary)
    dataloader = torch.utils.data.DataLoader(data_set,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             drop_last=False)
    return data_set, dataloader


def log_parameters(**kwargs):
    for key, value in kwargs.items():
        mlflow.log_param(str(key), value)


def main(_):
    data = dataset.read_lines(FLAGS.path_to_data)
    data_set = dataset.SeqDataSet(data, None)
    num_training_examples = int(FLAGS.train_validation_ratio * len(data))
    num_validation_examples = len(data) - num_training_examples
    training_dataset, validation_dataset = torch.utils.data.random_split(
        data_set, [num_training_examples, num_validation_examples])

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

    source_vocab = training_dataset.dataset.source_vocabulary

    # Create the temporary directory.
    temp_dir_path = tempfile.TemporaryDirectory('temp_output')

    logging.info('Number of training examples %i', len(training_dataset))
    logging.info('Number of validation examples %i', len(validation_dataset))

    mlflow.start_run()
    log_parameters(num_tokens_in_vocabulary=len(data_set.source_vocabulary),
                   dim_model=FLAGS.dim_model,
                   num_attention_heads=FLAGS.num_attention_heads,
                   dim_feedforward=FLAGS.dim_feedforward,
                   num_encoder_layers=FLAGS.num_encoder_layers,
                   learning_rate=FLAGS.learning_rate,
                   max_sequence_length=FLAGS.max_sequence_len,
                   path_to_data=FLAGS.path_to_data,
                   window_size=FLAGS.window_size,
                   latent_dim=FLAGS.latent_dim,
                   num_training_examples=num_training_examples,
                   num_validation_examples=num_validation_examples)

    with open(os.path.join(temp_dir_path.name, 'source_vocab.pickle'),
              'wb') as handle:
        pickle.dump(source_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    mlflow.log_artifact(os.path.join(temp_dir_path.name,
                                     'source_vocab.pickle'),
                        artifact_path='vocabulary')

    # Create the transformer.
    model = SeqTransformer(len(data_set.source_vocabulary), FLAGS.dim_model,
                           FLAGS.num_attention_heads, FLAGS.dim_feedforward,
                           FLAGS.num_encoder_layers, FLAGS.learning_rate,
                           FLAGS.max_sequence_len, FLAGS.window_size,
                           FLAGS.latent_dim, FLAGS.dropout)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=FLAGS.early_stopping_patience,
        mode='min')

    all_callbacks = [
        early_stopping_callback,
        MLFlow(save_dir=temp_dir_path.name)
    ]

    accelerator = 'gpu' if FLAGS.use_gpu else 'cpu'
    trainer = pl.Trainer(max_epochs=FLAGS.max_epochs,
                         accelerator=accelerator,
                         callbacks=all_callbacks)

    trainer.fit(model, training_data_loader, validation_data_loader)


if __name__ == '__main__':
    app.run(main)
