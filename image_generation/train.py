import os
import random

from absl import app
from absl import flags
from absl import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import dataset
from transformer import SeqTransformer

import mlflow

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 128, 'Size of the mini batches.')

flags.DEFINE_integer('max_epochs', 200, 'Number of epochs to train.')

# The transformer dim is the number of expected features in the encoder/decoder
# inputs.
flags.DEFINE_integer('dim_model', 64, 'Dimension of the transformer.')

flags.DEFINE_integer('num_attention_heads', 4,
                     'Number of attention heads in each layer.')

flags.DEFINE_integer('num_encoder_layers', 2, 'Number of encoder layers.')

flags.DEFINE_integer('num_decoder_layers', 2, 'Number of decoder layers.')

flags.DEFINE_integer(
    'dim_feedforward', 64,
    'Dimension of the feed-forward network in the encoder layers.')

flags.DEFINE_float('learning_rate', 1e-3, 'The learning rate of the optimizer.')

flags.DEFINE_integer(
    'max_sequence_len', 200,
    'Maximum length of the target sequence when using the model for inference')

flags.DEFINE_string('path_to_data', None,
                    'Path to the file containing the training data.')

flags.DEFINE_float(
    'train_validation_ratio', 0.9,
    'When no path to validation data is given we use a part of the training'
    'dataset for validation.')

flags.DEFINE_integer('num_cpus_per_worker', 1,
                     'The number of cpus for each worker.')

flags.DEFINE_boolean('use_gpu', False, 'Controls if the gpu is used.')

flags.DEFINE_integer(
    'early_stopping_patience', 10,
    'How many epochs to wait for improvement before stopping.')

flags.DEFINE_string('output_dir', 'output',
                    'Directory to save the outputs of the experiment.')

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
    training_data, validation_data = torch.utils.data.random_split(
        data,
        [num_training_examples, num_validation_examples],
        generator=torch.Generator().manual_seed(42))

    training_dataset = dataset.SeqDataSet(training_data, data_set.source_vocabulary)
    training_data_loader = torch.utils.data.DataLoader(training_dataset,
                                             batch_size=FLAGS.batch_size,
                                             shuffle=True,
                                                       drop_last=True)

    validation_dataset = dataset.SeqDataSet(validation_data, data_set.source_vocabulary)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset,
                                             batch_size=FLAGS.batch_size,
                                             shuffle=False,
                                             drop_last=False)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    logging.info('Number of training examples %i', len(training_dataset))
    logging.info('Number of validation examples %i', len(validation_dataset))

    mlflow.start_run()
    log_parameters(num_tokens_in_vocabulary=len(data_set.source_vocabulary),
                   dim_model=FLAGS.dim_model,
                   num_attention_heads=FLAGS.num_attention_heads,
                   dim_feedforward=FLAGS.dim_feedforward,
                   num_encoder_layers=FLAGS.num_encoder_layers,
                   learning_rate=FLAGS.learning_rate,
                   max_sequence_length=FLAGS.max_sequence_len)

    # Create the transformer.
    model = SeqTransformer(
        len(data_set.source_vocabulary),
        FLAGS.dim_model,
        FLAGS.num_attention_heads,
        FLAGS.dim_feedforward,
        FLAGS.num_encoder_layers,
        FLAGS.learning_rate,
        FLAGS.max_sequence_len)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=FLAGS.early_stopping_patience,
        mode='min')

    model_checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(
        FLAGS.output_dir, 'model_checkpoints'),
                                                monitor='val_loss',
                                                save_top_k=1)

    all_callbacks = [early_stopping_callback, model_checkpoint_callback]

    accelerator = 'gpu' if FLAGS.use_gpu else None
    trainer = pl.Trainer(max_epochs=FLAGS.max_epochs,
                             accelerator=accelerator,
                             callbacks=all_callbacks)

    trainer.fit(model, training_data_loader, validation_data_loader)


if __name__ == '__main__':
    app.run(main)
