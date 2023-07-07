"""Pytorch modules."""
import torch
from torch import nn


class ResBlock(nn.Module):

    def __init__(self, in_channel, channel, kernel_size=3, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel,
                      channel,
                      kernel_size=kernel_size,
                      padding=1,
                      stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class VQVAE(nn.Module):
    """Complete VQVAE"""

    def __init__(self, encoder, quantizer, decoder):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

    def forward(self, x):
        encoded_x = self.encoder(x)
        quantized_x, codebook_indices, loss = self.quantizer(encoded_x)
        decoded_x = self.decoder(quantized_x)
        return decoded_x, quantized_x, codebook_indices, loss


class Decoder(nn.Module):
    """The decoder module."""

    def __init__(self, channels, out_channels):
        super().__init__()

        self.convs = nn.Sequential(
            nn.ConvTranspose2d(channels,
                               channels,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.ConvTranspose2d(channels,
                               out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1))

        self.res_blocks = nn.Sequential(ResBlock(channels, channels),
                                        ResBlock(channels, channels))

        self.args = {'channels': channels, 'out_channels': out_channels}

    def forward(self, x):
        # input shape: B, H, W, C
        # reshape to: B, C, H, W
        x = x.permute(0, 3, 1, 2)
        x = self.res_blocks(x)
        x = self.convs(x)
        # Reshape again to return data (B, H, W, C)
        return nn.ReLU()(x).permute(0, 2, 3, 1)


class Encoder(nn.Module):
    """The encoder as it is presented in the paper."""

    def __init__(self, in_channels, channels):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels,
                      channels,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1))

        self.res_blocks = nn.Sequential(ResBlock(channels, channels),
                                        ResBlock(channels, channels))

        self.args = {'in_channels': in_channels, 'channels': channels}

    def forward(self, x):
        # input shape: B, H, W, C
        # reshape to: B, C, H, W
        x = x.permute(0, 3, 1, 2)
        x = self.convs(x)
        x = self.res_blocks(x)
        # Reshape again to return data (B, H, W, C)
        return nn.ReLU()(x).permute(0, 2, 3, 1)


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2

    Implementation taken and adapted from:
    https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.args = {'n_e': n_e, 'e_dim': e_dim, 'beta': beta}

    def forward(self, z):
        z_flattened = z.view(-1, self.e_dim)

        # matrix with distance to the encodings. (B, W * H, n_e)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + \
            self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, min_encoding_indices, loss

    def get_codebook_arrays(self, indices, shape):
        return self.embedding(indices).view(shape)
