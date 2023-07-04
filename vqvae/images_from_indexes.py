"""Uses the model to generate images from codebook indexes."""
import os

from absl import app
from absl import flags

import numpy as np
import torch
import matplotlib.pyplot as plt

import modules

FLAGS = flags.FLAGS

flags.DEFINE_string('indexes_path', None,
                    'Path to a txt file containing codebook indexes.')

flags.DEFINE_string('model_dir', None, 'The path to the VQVAE model.')

flags.DEFINE_integer('sequence_length', 256, 'The length of the sequence.')


def replace_token_by_most_commun(token, x):
    most_commun = max(set(x) - set(['<EOS>', '<SOS>']), key=x.count)
    return [most_commun if value == token else value for value in x]


def load_indices_from_file(file, required_length):
    with open(file, 'r') as file:
        data = file.readlines()

    data = [line.split(' ') for line in data]
    data = filter(lambda x: len(x) >= required_length, data)
    data = [line[:required_length] for line in data]
    data = map(lambda x: replace_token_by_most_commun('<EOS>', x), data)
    data = map(lambda x: replace_token_by_most_commun('<SOS>', x), data)
    data = [list(map(int, sequence)) for sequence in data]
    return data


def main(_):
    quantizer_path = os.path.join(FLAGS.model_dir, 'quantizer.pt')
    decoder_path = os.path.join(FLAGS.model_dir, 'decoder.pt')

    quantizer_weights, quantizer_args = torch.load(quantizer_path)
    decoder_weights, decoder_args = torch.load(decoder_path)

    quantizer = modules.VectorQuantizer(**quantizer_args)
    decoder = modules.Decoder(**decoder_args)

    quantizer.load_state_dict(quantizer_weights)
    decoder.load_state_dict(decoder_weights)

    num_channels = decoder_args['channels']
    latent_space_dim = int(np.sqrt(FLAGS.sequence_length))

    device = torch.device('cuda:0')
    quantizer.to(device)
    decoder.to(device)

    indexes = load_indices_from_file(FLAGS.indexes_path, FLAGS.sequence_length)

    for index in indexes:
        index = torch.tensor(index, dtype=torch.int, device=device)
        index = index.unsqueeze(0)

        codebook_arrays = quantizer.get_codebook_arrays(
            index, (1, latent_space_dim, latent_space_dim, num_channels))

        with torch.no_grad():
            output = decoder(codebook_arrays).cpu().numpy()[0]

        plt.imshow(output)
        plt.show()


if __name__ == '__main__':
    app.run(main)
