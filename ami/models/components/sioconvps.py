# SioConv with Parallel Scan  (PS)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .stacked_hidden_state import StackedHiddenState


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight: nn.Parameter | None = nn.Parameter(torch.ones(dim)) if self.elementwise_affine else None

    def forward(self, x: Tensor) -> Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            output = output * self.weight
        return output


class FFNSwiGLU(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim_ff_hidden)
        self.fc_act = nn.Linear(dim, dim_ff_hidden)
        self.fc_out = nn.Linear(dim_ff_hidden, dim)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x) * self.act(self.fc_act(x))
        x = self.fc_out(x)
        return x


class SioConvPSLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.fc_ln_z = nn.Linear(dim, dim)
        self.fc_y = nn.Linear(dim, dim)
        self.fc_y_act = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc_dt = nn.Linear(dim, dim)

    # (batch, len, dim), (batch, dim) -> (batch, len, dim), (batch, len, dim)
    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        ln_z = -F.softplus(-self.fc_ln_z(x))  # (batch, len, dim)

        ln_da = -F.softplus(-self.fc_dt(x))  # (batch, len, dim)
        ln_z_da = ln_z + ln_da
        ln_o_da = -F.softplus(self.fc_dt(x))  # (batch, len, dim)
        ln_o_da_cumsum = torch.cumsum(ln_o_da, dim=1)

        ln_z_da_ln_o_da_cumsum = ln_z_da - ln_o_da_cumsum  # (batch, len, dim)
        logcumsumexp_ln_z_da_ln_o_da_cumsum = torch.logcumsumexp(ln_z_da_ln_o_da_cumsum, dim=1)  # (batch, len, dim)

        h_inner_chunk = torch.exp(logcumsumexp_ln_z_da_ln_o_da_cumsum + ln_o_da_cumsum)  # (batch, len, dim)

        h_cross_chunk = torch.einsum("bld,bd->bld", torch.exp(ln_o_da_cumsum), hidden)  # (batch, len, dim)

        h = h_inner_chunk + h_cross_chunk

        y = self.fc_y(h) * self.act(self.fc_y_act(x))
        return y, h


class SioConvPSBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, dropout: float):
        super().__init__()
        self.sioconv = SioConvPSLayer(dim)
        self.ffn = FFNSwiGLU(dim, dim_ff_hidden)
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


class SioConvPS(StackedHiddenState):
    def __init__(self, depth: int, dim: int, dim_ff_hidden: int, dropout: float):
        super().__init__(nn.ModuleList([SioConvPSBlock(dim, dim_ff_hidden, dropout) for _ in range(depth)]))
