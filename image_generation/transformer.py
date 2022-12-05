import torch
from torch import nn
import pytorch_lightning as pl

import math

from layers import PositionalEncoding


class SeqTransformer(pl.LightningModule):

    def __init__(self,
                 ntoken,
                 d_model,
                 nhead,
                 d_hid,
                 nlayers,
                 learning_rate,
                 max_sequence_len,
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

        self.learning_rate = learning_rate

    def forward(self, source):
        # Transpose because we are working with batch first.
        source = torch.transpose(source, 0, 1)

        source = self.embedding(source) * math.sqrt(self.d_model)
        source = self.pos_encoder(source)
        # Compute the mask
        src_mask = self.generate_square_subsequent_mask(
            source.shape[0]).type_as(source)
        output = self.transformer_encoder(source, src_mask)
        output = self.linear(output)
        output = self.final_activation(output)
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
        # print(f'batch_in shape = {batch_in.shape}')
        # print(f'batch_out shape = {batch_out.shape}')

        predictions = self.forward(batch_in)

        flattened_predictions = torch.reshape(
            predictions,
            (batch_in.shape[0] * batch_in.shape[1], self.n_tokens))
        # print(f'flattened_predictions shape = {flattened_predictions.shape}')
        # print(f'100th letter prediction = {torch.exp(flattened_predictions[100])}')
        # print(f'targer shape = {torch.reshape(batch_out, (batch_out.shape[0] * batch_out.shape[1], )).shape}')
        # print(f'first actual word = {batch_out[0, 100]}')
        # print(f'value of correct prediction = {torch.exp(flattened_predictions[100][batch_out[0, 100]])}')

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
                            stop_when_eos=True):
        return self.auto_regressive_generation(
            source, lambda x: torch.multinomial(torch.exp(x), 1),
            end_of_sequence_symbol, max_steps, stop_when_eos)

    def top_k_generation(self,
                         source,
                         k,
                         end_of_sequence_symbol,
                         max_steps,
                         stop_when_eos=True):

        def f(predictions):
            top_k = torch.topk(predictions, k).indices.double()
            return top_k[torch.randperm(top_k.shape[0])].view(
                top_k.size())[0:1]

        return self.auto_regressive_generation(source, f,
                                               end_of_sequence_symbol,
                                               max_steps, stop_when_eos)

    def auto_regressive_generation(self,
                                   source,
                                   choice_function,
                                   end_of_sequence_symbol,
                                   max_steps,
                                   stop_when_eos=True):
        output_so_far = torch.tensor([[source]]).to(device=self.device)

        for _ in range(max_steps):
            predictions = self.forward(output_so_far)
            next_word_predictions = predictions[0][-1]
            next_word = choice_function(next_word_predictions)
            output_so_far = torch.cat(
                (output_so_far, torch.tensor([[next_word]])), 1).to(torch.int).to(device=self.device)
            if next_word == end_of_sequence_symbol and stop_when_eos:
                break
        return output_so_far

    def generate_square_subsequent_mask(self, sz):
        """Generates a mask for teacher forcing."""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
