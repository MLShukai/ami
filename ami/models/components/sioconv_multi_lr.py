import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .sioconvps import RMSNorm, SioConvPSLayer
from .stacked_hidden_state import StackedHiddenState
from typing import Callable


class MultiLRMLP(nn.Module):
    def __init__(
        self, dim: int, dim_ff_hidden: int, lr_scale: float, weight_decay: float, activation: Callable[[Tensor], Tensor] = nn.SiLU()
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim_ff_hidden)
        self.fc2 = nn.Linear(dim_ff_hidden, dim)
        self.activation = activation

        self.lr_scale = torch.exp(torch.linspace(0, np.log(lr_scale), dim_ff_hidden))
        self.weight_decay = torch.exp(np.log(weight_decay) + torch.linspace(-np.log(lr_scale), 0, dim_ff_hidden))

        self.fc1.weight.register_hook(
            lambda grad: grad.mul(self.lr_scale[:, None]) + self.fc1.weight.mul(self.weight_decay[:, None])
        )
        self.fc1.bias.register_hook(lambda grad: grad.mul(self.lr_scale) + self.fc1.bias.mul(self.weight_decay))
        self.fc2.weight.register_hook(
            lambda grad: grad.mul(self.lr_scale[None, :]) + self.fc2.weight.mul(self.weight_decay[None, :])
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
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
