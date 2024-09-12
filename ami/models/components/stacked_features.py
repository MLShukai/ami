import math

import torch
import torch.nn as nn
from torch import Tensor


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
