import math

import torch
import torch.nn as nn
from torch import Tensor


class LerpStackedFeatures(nn.Module):
    """Linear interpolation along stack of features.

    Shape:
        - stacked_features: (S, N_in) | (B, S, N_in)

        Return shape: (N_out,) | (B, N_out)
    """

    def __init__(self, dim_in: int, dim_out: int, num_stack: int, num_head: int) -> None:
        super().__init__()
        assert dim_in % num_head == 0

        self.feature_linear_weight = nn.Parameter(torch.randn(num_stack, dim_in, dim_out))
        self.feature_linear_bias = nn.Parameter(torch.randn(num_stack, dim_out))
        self.logit_coef_proj = nn.Linear(num_stack * dim_in, num_stack)
        self.num_head = num_head
        self.norm = nn.InstanceNorm1d(num_head)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # From `nn.Linear` implementation.

        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.feature_linear_weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.feature_linear_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.feature_linear_bias, -bound, bound)

    def forward(self, stacked_features: Tensor) -> Tensor:
        is_batch = len(stacked_features.shape) == 3
        if not is_batch:
            stacked_features = stacked_features.unsqueeze(0)

        batch, n_stack, dim = stacked_features.shape
        stacked_features = self.norm(
            stacked_features.reshape(batch * n_stack, self.num_head, dim // self.num_head)
        ).view(batch, n_stack, dim)

        logit_coef = self.logit_coef_proj(stacked_features.reshape(batch, n_stack * dim))

        feature_linear = torch.einsum(
            "sio,bsi->bso", self.feature_linear_weight, stacked_features
        ) + self.feature_linear_bias.unsqueeze(0)

        out = torch.einsum("bs,bsi->bi", torch.softmax(logit_coef, dim=-1), feature_linear)

        if not is_batch:
            out = out.squeeze(0)
        return out


class ToStackedFeatures(nn.Module):
    """Convert input features to the stacked features.

    Shape:
        - input_features: (N_in, ) | (B, N_in)

        Return shape: (S, N_out) | (B, S, N_out)
    """

    def __init__(self, dim_in: int, dim_out: int, num_stack: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.randn(dim_in, num_stack, dim_out) * (dim_out**-0.5))
        self.bias = nn.Parameter(torch.empty(num_stack, dim_out) * (dim_out**-0.5))

    def forward(self, feature: Tensor) -> Tensor:
        no_batch = feature.ndim == 1
        if no_batch:
            feature = feature.unsqueeze(0)

        out = torch.einsum("bi,isj->bsj", feature, self.weight) + self.bias

        if no_batch:
            out = out.squeeze(0)
        return out
