"""Pytorch modules."""
import torch
from torch import nn
from torch.nn import functional as F

import distributed as dist_fn

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size=3, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, kernel_size=kernel_size, padding=1, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class Encoder(nn.Module):
    """The encoder as it is presented in the paper."""
    def __init__(self, in_channels, channels):
        super().__init__()


        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        )

        self.res_blocks = nn.Sequential(
            ResBlock(channels, channels),
            ResBlock(channels, channels)
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.res_blocks(x)
        return nn.ReLU()(x)

