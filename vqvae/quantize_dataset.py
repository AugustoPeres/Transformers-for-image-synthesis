"""Imports the model and the data. Encodes all images"""
import os

from absl import app
from absl import flags

import tempfile

import torch
import mlflow

import modules
import datasets

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_dir', None,
    'The path to the directory with encoder quantizer and decoder.')
flags.DEFINE_string('path_to_data', None, 'Path to the data.')

flags.DEFINE_string('run_id', None, 'The id of the run.')


def main(_):
    encoder_path = os.path.join(FLAGS.model_dir, 'encoder.pt')
    quantizer_path = os.path.join(FLAGS.model_dir, 'quantizer.pt')
    decoder_path = os.path.join(FLAGS.model_dir, 'decoder.pt')

    encoder_weights, encoder_args = torch.load(encoder_path)
    quantizer_weights, quantizer_args = torch.load(quantizer_path)
    decoder_weights, decoder_args = torch.load(decoder_path)

    encoder = modules.Encoder(**encoder_args)
    quantizer = modules.VectorQuantizer(**quantizer_args)
    decoder = modules.Decoder(**decoder_args)

    encoder.load_state_dict(encoder_weights)
    quantizer.load_state_dict(quantizer_weights)
    decoder.load_state_dict(decoder_weights)

    model = modules.VQVAE(encoder, quantizer, decoder)

    dataset = datasets.ImageDataset(FLAGS.path_to_data)

    # open the temporary file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt') as temp_file:
        with torch.no_grad():
            # Iterate over the dataset storing the codebook indexes in a
            # temporary text file and then logging that file to mlflow.
            for i in range(len(dataset)):
                image = dataset[i]['image']
                _, _, codebook_indices, _ = model(image.unsqueeze(0))
                codebook_indices = map(str, list(codebook_indices.numpy()))
                # Write the indeces seperated by a space.
                temp_file.write(' '.join(codebook_indices) + '\n')
            with mlflow.start_run(run_id=FLAGS.run_id):
                mlflow.log_artifact(temp_file.name)


if __name__ == '__main__':
    app.run(main)
