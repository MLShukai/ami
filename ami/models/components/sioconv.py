import einops
import numpy as np
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


class SioConvLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_head: int,
        a_init_range: tuple[float, float] = (1, 16),
        dt_init_range: tuple[float, float] = (0.001, 0.1),
    ):
        super().__init__()
        assert dim % num_head == 0, "dim must be multiple of num_head"
        self.dim = dim
        self.num_head = num_head
        self.fc_z = nn.Linear(dim, dim)
        self.fc_z_act = nn.Linear(dim, dim)
        self.fc_y = nn.Linear(dim, dim)
        self.fc_y_act = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.ln_a = nn.Parameter(torch.log(torch.empty(num_head).uniform_(a_init_range[0], a_init_range[1])))
        self.fc_dt = nn.Linear(dim, num_head)
        dt = torch.exp(torch.empty(num_head).uniform_(np.log(dt_init_range[0]), np.log(dt_init_range[1])))
        # inv_softplus_dt = torch.log(torch.exp(dt)-1) equals
        inv_softplus_dt = dt + torch.log(1 - torch.exp(-dt))
        self.fc_dt.bias = nn.Parameter(inv_softplus_dt)
        self.norm = nn.GroupNorm(num_head, num_head)
        self.is_refresh = True
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.fc_dt.weight, gain=1e-2)
        nn.init.xavier_normal_(self.fc_z.weight, gain=1e-2)
        nn.init.zeros_(self.fc_z.bias)
        nn.init.xavier_normal_(self.fc_z_act.weight, gain=1e-2)
        nn.init.zeros_(self.fc_z_act.bias)
        nn.init.xavier_normal_(self.fc_y.weight, gain=1e-2)
        nn.init.zeros_(self.fc_y.bias)
        nn.init.xavier_normal_(self.fc_y_act.weight, gain=1e-2)
        nn.init.zeros_(self.fc_y_act.bias)

    # (batch, len, dim), (batch, num_head, inner_dim) -> (batch, len, dim), (batch, len, num_head, inner_dim)
    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        batch = x.shape[0]
        len = x.shape[1]
        dim = x.shape[2]
        num_head = self.num_head
        inner_dim = dim // num_head

        z = (self.fc_z(x) * self.act(self.fc_z_act(x))).view(
            batch, len, num_head, inner_dim
        )  # (batch, len, num_head, inner_dim)

        ones = torch.ones(len, device=x.device, dtype=x.dtype)
        ones_fft = torch.fft.rfft(ones, n=len * 2)

        ln_da = -torch.exp(self.ln_a) * F.softplus(self.fc_dt(x))  # (batch, len, num_head)
        ln_da_masked = einops.repeat(ln_da, "b l h ->b l m h", m=len).tril(-1)  # (batch, len, len, num_head)
        ln_da_masked_fft = torch.fft.rfft(ln_da_masked, n=len * 2, dim=1)  # (batch, len, len, num_head)
        ln_da_masked_conv = torch.fft.irfft(torch.einsum("blmh,l->blmh", ln_da_masked_fft, ones_fft), dim=1).narrow(
            1, 0, len
        )  # (batch, len, len, num_head)
        da_masked_conv = torch.exp(ln_da_masked_conv).tril()  # (batch, len, len, num_head)

        h_inner_chunk = torch.einsum("blmh,bmhi->blhi", da_masked_conv, z)

        ln_da_fft = torch.fft.rfft(ln_da, n=len * 2, dim=1)
        ln_da_conv = torch.fft.irfft(torch.einsum("blh,l->blh", ln_da_fft, ones_fft), dim=1).narrow(
            1, 0, len
        )  # (batch, len, num_head)

        h_cross_chunk = torch.einsum("blh,bhi->blhi", torch.exp(ln_da_conv), hidden)

        h = h_inner_chunk + h_cross_chunk

        hidden_next = h

        h_norm = self.norm(h.reshape(batch * len, num_head, inner_dim)).view(batch, len, dim)
        y = self.fc_y(h_norm) * self.act(self.fc_y_act(x))
        return y, hidden_next


class ChunkWiseSioConvLayer(nn.Module):
    def __init__(self, dim: int, num_head: int, chunk_size: int):
        super().__init__()
        self.sioconv = SioConvLayer(dim, num_head)
        self.last_hidden = None
        self.last_hidden_init = nn.Parameter(torch.randn(num_head, dim // num_head))
        self.is_refresh = True
        self.dim = dim
        self.num_head = num_head
        self.chunk_size = chunk_size

    # (batch, len, dim), (batch, dim) -> (batch, len, dim), (batch, len, dim)
    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        batch = x.shape[0]
        len = x.shape[1]
        num_head = self.num_head
        dim = self.dim

        input_chunks = x.split(self.chunk_size, dim=1)
        hidden = hidden.view(batch, num_head, dim // num_head)
        output_chunks = []
        hidden_next_chunks = []
        for input_chunk in input_chunks:
            output_chunk, hidden_next_chunk = self.sioconv(input_chunk, hidden)
            output_chunks.append(output_chunk)
            hidden_next_chunks.append(hidden_next_chunk)
            hidden = hidden_next_chunk[:, -1, :, :]

        output = torch.cat(output_chunks, dim=1)
        hidden_next = torch.cat(hidden_next_chunks, dim=1).view(batch, len, dim)
        return output, hidden_next


class SioConvBlock(nn.Module):
    def __init__(self, dim: int, num_head: int, dim_ff_hidden: int, dropout: float, chunk_size: int):
        super().__init__()
        self.sioconv = ChunkWiseSioConvLayer(dim, num_head, chunk_size)
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


class SioConv(StackedHiddenState):
    def __init__(self, depth: int, dim: int, num_head: int, dim_ff_hidden: int, dropout: float, chunk_size: int):
        super().__init__(
            nn.ModuleList([SioConvBlock(dim, num_head, dim_ff_hidden, dropout, chunk_size) for _ in range(depth)])
        )
