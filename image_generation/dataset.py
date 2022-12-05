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
    # Creates an object v such that
    # v['a'] = some_index and
    # v['word out of vocab'] = -1
    vocab = torchtext.vocab.vocab(counter)
    vocab.set_default_index(-1)
    return vocab


class SeqDataSet(torch.utils.data.Dataset):
    """Pytorch data set containing synthetic data for seq2seq problems.

    This data set contains examples of sequence pairs:

    (RWFLDEGNPGQQL, AATTGCATTCGAAGATCAACTGGACCAACTATGATGCGAA)

    The first (input) sequence represents a sequence of amino-acids and the
    second (target) sequence represents a sequence of nucleotides.
    """

    def __init__(self, seq_data, source_vocab):
        super(SeqDataSet).__init__()

        data = [source.split(' ') for source in seq_data]

        # Now we add the start of sequence (SOS) and end of sequence (EOS)
        # markers to the target sequence.
        data = [['<SOS>'] + source + ['<EOS>'] for source in data]

        # Computing the size of the longest sequences.
        size_of_longest_source_sequence = max(map(len, data))

        source_padding = ['PAD'] * size_of_longest_source_sequence

        # Padding so that every sequence has the same size.
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
