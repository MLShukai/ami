import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .stacked_hidden_state import StackedHiddenState


class FFN(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim_ff_hidden, bias=True)
        self.linear_2 = nn.Linear(dim_ff_hidden, dim, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class SConvLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.phazor_init = nn.Parameter(torch.view_as_real(torch.randn(dim, dtype=torch.cfloat)))  # log(-log(gamma))
        self.phazor = nn.Parameter(
            torch.view_as_real(torch.exp(2.0j * np.pi * torch.arange(dim) / dim) * torch.abs(torch.randn(dim)))
        )

    # ((batch, len, dim),(batch, dim)) -> ((batch, len, dim), (batch, len, dim))
    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        batch = x.shape[0]
        len = x.shape[1]
        phazor = torch.view_as_complex(self.phazor)
        phazor = torch.exp(-phazor.real * phazor.real - phazor.imag * phazor.imag) * torch.exp(1.0j * phazor.angle())
        phazor_progression = torch.pow(
            phazor.unsqueeze(0), torch.arange(len, device=x.device).unsqueeze(1)
        )  # (len, dim)
        filter = phazor_progression * torch.view_as_complex(self.phazor_init).unsqueeze(0)
        filter_fft = torch.fft.fft(filter, n=len * 2, dim=0)  # (len*2, dim)
        x_fft = torch.fft.fft(x, n=len * 2, dim=1)  # (batch, len*2, dim)
        conv_filter_x = torch.fft.ifft(filter_fft.unsqueeze(0) * x_fft, dim=1).narrow(1, 0, len)  # (batch, len, dim)
        conv_with_past = conv_filter_x + hidden.detach().unsqueeze(1) * phazor_progression.unsqueeze(
            0
        ) * phazor.unsqueeze(0).unsqueeze(0)

        return conv_with_past.real, conv_with_past


class SconvBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, dropout: float):
        super().__init__()
        self.spiral_conv = SConvLayer(dim)
        self.ffn = FFN(dim, dim_ff_hidden)
        self.layer_norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        x_ = x
        y = x
        x = self.layer_norm(x)
        x, hidden = self.spiral_conv(x, hidden)
        y = self.fc(y)
        y = self.silu(y)
        x = x * y
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.layer_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_

        return x, hidden


class SConv(StackedHiddenState):
    def __init__(self, depth: int, dim: int, dim_ff_hidden: int, dropout: float):
        super().__init__(nn.ModuleList([SconvBlock(dim, dim_ff_hidden, dropout) for _ in range(depth)]))
