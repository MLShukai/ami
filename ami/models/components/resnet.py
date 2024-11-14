import torch.nn as nn
from torch import Tensor


class ResNetFF(nn.Module):
    def __init__(self, dim: int, dim_hidden: int, depth: int, activation: nn.Module = nn.GELU()):
        super().__init__()
        self.ff_list = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim_hidden), activation, nn.Linear(dim_hidden, dim))
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for ff in self.ff_list:
            x_ = x
            x = ff(x)
            x = x + x_
        return x
