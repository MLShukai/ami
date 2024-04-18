import torch.nn as nn
from torch import Tensor


class FullyConnectedValueHead(nn.Module):
    def __init__(self, dim_in: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim_in, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)
