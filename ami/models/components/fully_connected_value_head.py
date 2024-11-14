import torch.nn as nn
from torch import Tensor


class FullyConnectedValueHead(nn.Module):
    def __init__(self, dim_in: int, squeeze_value_dim: bool = False) -> None:
        super().__init__()
        self.fc = nn.Linear(dim_in, 1)
        self.squeeze_value_dim = squeeze_value_dim

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.fc(x)
        if self.squeeze_value_dim:
            out = out.squeeze(-1)
        return out
