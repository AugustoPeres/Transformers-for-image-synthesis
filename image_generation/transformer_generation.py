"""Uses the transformer to generate several sequences."""
import os
import pickle

from absl import app
from absl import flags
from absl import logging

import torch

import dataset
from transformer import SeqTransformer

import mlflow

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', None, 'The path for the model.')
flags.DEFINE_string('vocabulary_path', None, 'The path to the vocabulary.')

flags.DEFINE_integer(
    'max_sequence_len', 300,
    'Maximum length of the target sequence when using the model for inference')

flags.DEFINE_string('run_id', None, 'MLFLOW run id')

flags.DEFINE_integer('num_sequences_to_generate', 50,
                     'The number of sequences, by each method, to generate.')


def main(_):
    # Load the transfomer.
    model = SeqTransformer.load_from_checkpoint(FLAGS.model_path)

    with open(FLAGS.vocabulary_path, 'rb') as pickle_file:
        source_vocab = pickle.load(pickle_file)

    # Generate sequences by sampling without context.
    sequences = []
    # for temperature in [1, .8, .7, .6, .5]:
    for temperature in [1, .8, .5]:
        for _ in range(FLAGS.num_sequences_to_generate):
            with torch.no_grad():
                source = torch.tensor([[source_vocab['<SOS>']]])
                sequence = model.sampling_generation(
                    source,
                    source_vocab['<EOS>'],
                    FLAGS.max_sequence_len,
                    stop_when_eos=False,
                    temperature=temperature).cpu().numpy()[0]

            # Recover the codebook indexes from the token indexes.
            sequence = source_vocab.lookup_tokens(sequence)
            print(sequence)
            sequences.append(' '.join(map(str, sequence)) + '\n')

    # Create a file with the generated sequences and log them.
    with open('generated_sequences.txt', 'w', encoding='utf-8') as f:
        f.writelines(sequences)
    with mlflow.start_run(run_id=FLAGS.run_id):
        mlflow.log_artifact('generated_sequences.txt',
                            artifact_path='generated_sequences')


if __name__ == '__main__':
    app.run(main)
