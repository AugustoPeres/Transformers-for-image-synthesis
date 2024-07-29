from collections import Counter
import itertools

import torch
import torchtext

import warnings


def read_lines(path_to_txt):
    # Read the whole file.
    with open(path_to_txt, encoding='utf-8') as file:
        data = file.read().splitlines()
    return data


def _make_vocab(tokens):
    """Creates the mappings referenced in the doc string.

    Args:
      tokens:
        A list of tokens, e.g, ['a', 'a', 'b']
    """
    counter = Counter(tokens)
    vocab = torchtext.vocab.vocab(counter)
    vocab.set_default_index(-1)
    return vocab


class SeqDataSet(torch.utils.data.Dataset):
    """Torch dataset for sequence modeling. """

    def __init__(self, seq_data, source_vocab):
        """
        Args:
          seq_data: List of 'Token_1 Token_1' type strings.
          source_vocab: TorchText vocabulary

        """
        super(SeqDataSet).__init__()

        data = [source.split(' ') for source in seq_data]

        data = [['<SOS>'] + source + ['<EOS>'] for source in data]

        size_of_longest_source_sequence = max(map(len, data))

        source_padding = ['PAD'] * size_of_longest_source_sequence

        data = [(source + source_padding)[:size_of_longest_source_sequence]
                for source in data]

        self.data = data

        self.source_vocabulary = _make_vocab(itertools.chain(
            *list(self.data))) if source_vocab is None else source_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        sequence_as_indexes = [self.source_vocabulary[e] for e in sequence]

        if -1 in sequence_as_indexes:
            warnings.warn('OUT OF VOCABULARY WORD FOUND IN DATA.')
        return torch.tensor(sequence_as_indexes)
