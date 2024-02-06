from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal

from .small_conv_net import SmallConvNet
from .small_deconv_net import SmallDeconvNet


class Encoder(nn.Module):
    def __init__(self, base_model: nn.Module, min_stddev=1e-7) -> None:
        """Construct encoder for VAE. output channel size of the `base_model`
        is twice the size of the latent space.

        Args:
            base_model (nn.Module): The base convolutional neural network model for encoding.
            min_stddev (float, optional): Small value added to stddev for preventing stddev from being zero. Defaults to 1e-7.
        """
        super().__init__()
        self.conv_net = base_model
        self.min_stddev = min_stddev

    def forward(self, x: Tensor):
        mu_sigma = self.conv_net(x)
        mu, sigma = torch.chunk(mu_sigma, chunks=2, dim=-1)
        sigma = torch.nn.functional.softplus(sigma) + self.min_stddev
        distribution = Normal(mu, sigma)
        return distribution


class Decoder(nn.Module):
    def __init__(self, base_model: nn.Module):
        """Construct decoder for VAE.

        Args:
            base_model (nn.Module): base_model (nn.Module): The base convolutional neural network model for decoding.
        """
        super().__init__()
        self.deconv_net = base_model

    def forward(self, z: Tensor):
        rec_img = self.deconv_net(z)
        return rec_img


class VAE(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        """Construct VAE.

        Args:
            encoder (Encoder): The encoder for encoding input data.
            decoder (Decoder): The decoder for decoding latent variable.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z_dist: Normal = self.encoder(x)
        z_sampled = z_dist.rsample()
        x_reconstructed = self.decoder(z_sampled)
        return x_reconstructed, z_dist
