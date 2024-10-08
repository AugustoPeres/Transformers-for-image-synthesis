import torch
from torch import nn
import pytorch_lightning as pl

import math

from layers import PositionalEncoding


def f(n, latent_space_dim):
    return (n // latent_space_dim, n % latent_space_dim)


def g(i, j, latent_space_dim):
    return i * latent_space_dim + j


def make_attention_mask(seq_size, w, k):
    return [
        make_attention_mask_line(seq_size, i, w, k) for i in range(seq_size)
    ]


def make_attention_mask_line(seq_size, index, w, k):
    # Positions with -inf are not allowed to attend.
    i, j = f(index, k)
    attenting_to = [(i, j)]
    for vertical in range(w):
        allowed_range = range(-w, w) if vertical > 0 else range(-w, 0)
        for horizontal in allowed_range:
            if (j + horizontal >= 0 and j + horizontal < k
                    and i - vertical >= 0):
                attenting_to.append((i - vertical, j + horizontal))

    attentding_to_flat = []
    aux = lambda i: float('-inf') if f(i, k) not in attenting_to else 0
    return [aux(i) for i in range(seq_size)]


class SeqTransformer(pl.LightningModule):

    def __init__(self,
                 ntoken,
                 d_model,
                 nhead,
                 d_hid,
                 nlayers,
                 learning_rate,
                 max_sequence_len,
                 window_size,
                 latent_dim,
                 dropout=0.5):
        super().__init__()
        self.save_hyperparameters()

        self.pos_encoder = PositionalEncoding(d_model,
                                              dropout,
                                              max_len=max_sequence_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid,
                                                    dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.final_activation = nn.LogSoftmax(dim=2)

        self.n_tokens = ntoken

        self.window_size = window_size
        self.latent_dim = latent_dim

        self.learning_rate = learning_rate

    def forward(self, source, temperature=1):
        # Transpose because we are working with batch first.
        source = torch.transpose(source, 0, 1)

        source = self.embedding(source) * math.sqrt(self.d_model)
        source = self.pos_encoder(source)
        # Compute the mask
        src_mask = self.generate_square_subsequent_mask(
            source.shape[0]).type_as(source)
        slidding_window_mask = torch.tensor(
            make_attention_mask(source.shape[0], self.window_size,
                                self.latent_dim)).type_as(source)
        src_mask += slidding_window_mask
        output = self.transformer_encoder(source, src_mask)
        output = self.linear(output)
        output = self.final_activation(output / temperature)
        return torch.transpose(output, 0, 1)

    def training_step(self, batch, _):
        training_loss = self._compute_loss(batch)
        self.log('loss',
                 training_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        return {'loss': training_loss}

    def validation_step(self, batch, _):
        """Equal to training step but used for validation."""
        val_loss = self._compute_loss(batch)
        # Log the validation loss to the progress bar.
        self.log('val_loss',
                 val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        return {'val_loss': val_loss}

    def _compute_loss(self, batch):
        # Shift the inputs for teacher forcing. For example:
        # >>> batch = [0, 1, 2, 3, 4]
        # >>> batch_in = batch[:, :-1] = [0, 1, 2, 3]
        # >>> batch_out = batch[:, 1:] = [1, 2, 3, 4]
        # Then, self.forward(source, batch_in) is an array of shape
        # [4, size_target_vocobalary], e.g, [[0.6, 0.1, 0.1, 0.1, 0.1], ...]
        # where each element corresponds to the probabilities of the words
        # in target_out.
        batch_in = batch[:, :-1]
        batch_out = batch[:, 1:]

        predictions = self.forward(batch_in)

        flattened_predictions = torch.reshape(
            predictions,
            (batch_in.shape[0] * batch_in.shape[1], self.n_tokens))
        loss = nn.NLLLoss()(
            flattened_predictions,
            torch.reshape(batch_out,
                          (batch_out.shape[0] * batch_out.shape[1], )))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def sampling_generation(self,
                            source,
                            end_of_sequence_symbol,
                            max_steps,
                            stop_when_eos=True,
                            temperature=1):
        return self.auto_regressive_generation(
            source, lambda x: torch.multinomial(torch.exp(x), 1),
            end_of_sequence_symbol, max_steps, temperature, stop_when_eos)

    def top_k_generation(self,
                         source,
                         k,
                         end_of_sequence_symbol,
                         max_steps,
                         stop_when_eos=True):

        def f(predictions):
            values, indices = torch.topk(predictions, k)
            sampled = torch.multinomial(values / torch.sum(values), 1)
            return indices[sampled]

        return self.auto_regressive_generation(source,
                                               f,
                                               end_of_sequence_symbol,
                                               max_steps,
                                               stop_when_eos=stop_when_eos)

    def auto_regressive_generation(self,
                                   source,
                                   choice_function,
                                   end_of_sequence_symbol,
                                   max_steps,
                                   temperature=1,
                                   stop_when_eos=True):
        output_so_far = torch.tensor([[source]]).to(device=self.device)

        for _ in range(max_steps):
            predictions = self.forward(output_so_far, temperature=temperature)
            next_word_predictions = predictions[0][-1]
            next_word = choice_function(next_word_predictions)
            output_so_far = torch.cat(
                (output_so_far, torch.tensor([[next_word]
                                              ]).to(device=self.device)),
                1).to(torch.int).to(device=self.device)
            if next_word == end_of_sequence_symbol and stop_when_eos:
                break
        return output_so_far

    def generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
