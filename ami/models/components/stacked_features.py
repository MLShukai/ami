import math

import torch
import torch.nn as nn
from torch import Tensor

from .mixture_desity_network import NormalMixture


class LerpStackedFeatures(nn.Module):
    """Linear interpolation along stack of features.

    Shape:
        - stacked_features: (*, S, N_in)

        Return shape: (*, N_out)
    """

    def __init__(self, dim_in: int, dim_out: int, num_stack: int) -> None:
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_stack = num_stack

        self.feature_linear_weight = nn.Parameter(torch.randn(num_stack, dim_in, dim_out) * (dim_out**-0.5))
        self.feature_linear_bias = nn.Parameter(torch.randn(num_stack, dim_out) * (dim_out**-0.5))
        self.logit_coef_proj = nn.Linear(num_stack * dim_in, num_stack)

    def forward(self, stacked_features: Tensor) -> Tensor:
        no_batch = len(stacked_features.shape) == 2
        if no_batch:
            stacked_features = stacked_features.unsqueeze(0)

        batch_shape = stacked_features.shape[:-2]
        n_stack, dim = stacked_features.shape[-2:]
        stacked_features = stacked_features.reshape(-1, n_stack, dim)
        batch = stacked_features.size(0)

        logit_coef = self.logit_coef_proj(stacked_features.reshape(batch, n_stack * dim))

        feature_linear = torch.einsum(
            "sio,bsi->bso", self.feature_linear_weight, stacked_features
        ) + self.feature_linear_bias.unsqueeze(0)

        out = torch.einsum("bs,bsi->bi", torch.softmax(logit_coef, dim=-1), feature_linear)

        out = out.reshape(*batch_shape, -1)

        if no_batch:
            out = out.squeeze(0)
        return out


class ToStackedFeatures(nn.Module):
    """Convert input features to the stacked features.

    Shape:
        - input_features: (*, N_in)

        Return shape: (*, S, N_out)
    """

    def __init__(self, dim_in: int, dim_out: int, num_stack: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.randn(dim_in, num_stack, dim_out) * (dim_out**-0.5))
        self.bias = nn.Parameter(torch.randn(num_stack, dim_out) * (dim_out**-0.5))

        self.num_stack = num_stack

    def forward(self, feature: Tensor) -> Tensor:
        no_batch = feature.ndim == 1
        if no_batch:
            feature = feature.unsqueeze(0)
        batch_shape = feature.shape[:-1]
        feature = feature.reshape(-1, feature.size(-1))

        out = torch.einsum("bi,isj->bsj", feature, self.weight) + self.bias
        out = out.reshape(*batch_shape, *out.shape[-2:])

        if no_batch:
            out = out.squeeze(0)
        return out


class ToStackedFeaturesMDN(nn.Module):
    """Convert input features to stacked features using Mixture Density
    Network.

    This module combines ToStackedFeatures with a Mixture Density Network (MDN)
    to produce a probabilistic representation of stacked features.

    Shape:
        - input_features: (*, N_in)

        Return shape: NormalMixture with parameters:
            - logits: (*, num_components)
            - mu: (*, S, N_out, num_components)
            - sigma: (*, S, N_out, num_components)
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_stack: int,
        num_components: int,
        eps: float = 1e-5,
        squeeze_feature_dim: bool = False,
    ) -> None:
        super().__init__()

        self.to_stacked_features = ToStackedFeatures(dim_in, dim_out, num_stack)
        self.mu_layers = nn.ModuleList(nn.Linear(dim_out, dim_out) for _ in range(num_components))
        self.logvar_layers = nn.ModuleList(nn.Linear(dim_out, dim_out) for _ in range(num_components))
        self.logits_layer = nn.Linear(dim_in, num_components)
        self.eps = eps
        self.squeeze_feature_dim = squeeze_feature_dim

        self.init_weights()

    def init_weights(self) -> None:
        layer: nn.Linear
        for layer in self.logvar_layers:
            nn.init.normal_(layer.weight, 0.0, 0.01)
            nn.init.zeros_(layer.bias)

        nn.init.normal_(self.logits_layer.weight, 0.0, 0.01)
        nn.init.zeros_(self.logits_layer.bias)

    def forward(self, feature: Tensor) -> NormalMixture:
        logits = self.logits_layer(feature)
        out = self.to_stacked_features(feature)
        mu = torch.stack([lyr(out) for lyr in self.mu_layers], dim=-1)
        logvar = torch.stack([lyr(out) for lyr in self.logvar_layers], dim=-1)
        sigma = torch.exp(0.5 * logvar) + self.eps

        return NormalMixture(logits, mu, sigma)
