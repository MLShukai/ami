import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .sioconvps import RMSNorm, SioConvPSLayer
from .stacked_hidden_state import StackedHiddenState


class MultiLRMLP(nn.Module):
    def __init__(
        self, dim: int, dim_ff_hidden: int, lr_scale: float, weight_decay: float, activation: nn.Module = nn.SiLU()
    ):
        super().__init__()
        self.fc1_weight = nn.Parameter(torch.randn(dim_ff_hidden, dim))
        self.fc1_bias = nn.Parameter(torch.zeros(dim_ff_hidden))
        self.fc2_weight = nn.Parameter(torch.randn(dim, dim_ff_hidden))
        self.fc2_bias = nn.Parameter(torch.zeros(dim))
        self.activation = activation

        self.lr_scale = torch.exp(torch.linspace(0, np.log(lr_scale), dim_ff_hidden))
        self.weight_decay = torch.exp(np.log(weight_decay) + torch.linspace(-np.log(lr_scale), 0, dim_ff_hidden))

        self.fc1_weight.register_hook(
            lambda grad: grad.mul_(self.lr_scale[:, None]) + self.fc1_weight.mul_(self.weight_decay[:, None])
        )
        self.fc1_bias.register_hook(lambda grad: grad.mul_(self.lr_scale) + self.fc1_bias.mul_(self.weight_decay))
        self.fc2_weight.register_hook(
            lambda grad: grad.mul_(self.lr_scale[None, :]) + self.fc2_weight.mul_(self.weight_decay[None, :])
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.fc1_weight, self.fc1_bias)
        x = self.activation(x)
        x = F.linear(x, self.fc2_weight, self.fc2_bias)
        return x


class SioConvMultiLRBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, dropout: float, lr_scale: float, weight_decay: float):
        super().__init__()
        self.sioconv = SioConvPSLayer(dim)
        self.ffn = MultiLRMLP(dim, dim_ff_hidden, lr_scale, weight_decay)
        self.norm_sioconv = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        x_ = x
        x = self.norm_sioconv(x)
        x, hidden = self.sioconv(x, hidden)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x, hidden


class SioConvMultiLR(StackedHiddenState):
    def __init__(self, depth: int, dim: int, dim_ff_hidden: int, dropout: float, lr_scale: float, weight_decay: float):
        super().__init__(
            nn.ModuleList(
                [SioConvMultiLRBlock(dim, dim_ff_hidden, dropout, lr_scale, weight_decay) for _ in range(depth)]
            )
        )
