from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal

from .components.small_conv_net import SmallConvNet
from .components.small_deconv_net import SmallDeconvNet
from .model_wrapper import ModelWrapper


class Encoder(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: Tensor) -> Normal:
        raise NotImplementedError


class Decoder(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Conv2dEncoder(Encoder):
    def __init__(self, height: int, width: int, channels: int, latent_dim: int, do_batchnorm: bool = False) -> None:
        super().__init__()
        self.conv_net = SmallConvNet(height, width, channels, latent_dim, do_batchnorm=do_batchnorm)
        self.linear_mu = nn.Linear(latent_dim, latent_dim)
        self.linear_sigma = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: Tensor) -> Normal:
        latent = self.conv_net(x)
        mu = self.linear_mu(latent)
        sigma = self.linear_sigma(latent)
        sigma = torch.nn.functional.softplus(sigma) + 1e-7
        distribution = Normal(mu, sigma)
        return distribution


class Conv2dDecoder(Decoder):
    def __init__(self, height: int, width: int, channels: int, latent_dim: int, do_batchnorm: bool = False) -> None:
        super().__init__()
        self.deconv_net = SmallDeconvNet(height, width, channels, latent_dim, do_batchnorm=do_batchnorm)

    def forward(self, z: Tensor) -> Tensor:
        rec_img: Tensor = self.deconv_net(z)
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

    def forward(self, x: Tensor) -> tuple[Tensor, Normal]:
        z_dist: Normal = self.encoder(x)
        z_sampled = z_dist.rsample()
        x_reconstructed = self.decoder(z_sampled)
        return x_reconstructed, z_dist


def encoder_infer(wrapper: ModelWrapper[Encoder], x: torch.Tensor) -> torch.Tensor:
    z: torch.Tensor = wrapper.model.forward(x.to(wrapper.device)).loc
    return z
