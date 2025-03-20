import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from .sioconvps import RMSNorm, SioConvPSLayer
from .stacked_hidden_state import StackedHiddenState

class MultiLRMLP(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, num_head: int, lr_scale: float, weight_decay: float, activation: nn.Module=nn.SiLU()):
        super().__init__()
        assert dim_ff_hidden % num_head == 0
        head_dim = dim_ff_hidden // num_head
        self.fc1_weight = nn.Parameter(torch.randn(dim_ff_hidden, dim))
        self.fc1_bias = nn.Parameter(torch.zeros(dim_ff_hidden))
        self.fc2_weight = nn.Parameter(torch.randn(dim, dim_ff_hidden))
        self.fc2_bias = nn.Parameter(torch.zeros(dim))
        self.activation = activation

        self.lr_scale = torch.exp(torch.linspace(0, np.log(lr_scale), num_head))
        self.weight_decay = torch.exp(np.log(weight_decay) + torch.linspace(-np.log(lr_scale), 0, num_head))

        self.fc1_weight.register_hook(lambda grad: (
            grad.view(num_head, head_dim, dim).mul_(self.lr_scale[:,None,None]) +
            self.fc1_weight.view(num_head, head_dim, dim).mul_(self.weight_decay[:,None,None]))
        .view(dim_ff_hidden, dim))
        self.fc1_bias.register_hook(lambda grad: (
            grad.view(num_head, head_dim).mul_(self.lr_scale[:,None]) +
            self.fc1_bias.view(num_head, head_dim).mul_(self.weight_decay[:,None]))
        .view(dim_ff_hidden))
        self.fc2_weight.register_hook(lambda grad: (
            grad.view(dim, num_head, head_dim).mul_(self.lr_scale[None,:,None]) +
            self.fc2_weight.view(dim, num_head, head_dim).mul_(self.weight_decay[None,:,None]))
        .view(dim, dim_ff_hidden))

    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.fc1_weight, self.fc1_bias)
        x = self.activation(x)
        x = F.linear(x, self.fc2_weight, self.fc2_bias)
        return x

class SioConvMultiLRBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, dropout: float, num_head: int, lr_scale: float, weight_decay: float):
        super().__init__()
        self.sioconv = SioConvPSLayer(dim)
        self.ffn = MultiLRMLP(dim, dim_ff_hidden, num_head, lr_scale, weight_decay)
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
    def __init__(self, depth: int, dim: int, dim_ff_hidden: int, dropout: float, num_head: int, lr_scale: float, weight_decay: float):
        super().__init__(nn.ModuleList([SioConvMultiLRBlock(dim, dim_ff_hidden, dropout, num_head, lr_scale, weight_decay) for _ in range(depth)]))
