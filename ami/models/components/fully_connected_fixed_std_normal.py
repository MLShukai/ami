import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

SHIFT_ZERO = 1.0 / math.sqrt(2.0 * math.pi)
SCALE_ONE = 1.0 / math.sqrt(2.0)


class FullyConnectedFixedStdNormal(nn.Module):
    """The layer which returns the normal distribution with fixed standard
    deviation."""

    def __init__(self, in_dim: int, out_dim: int, std: float = SHIFT_ZERO, normal_cls: type[Normal] = Normal) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.std = std
        self.normal_cls = normal_cls

    def forward(self, x: Tensor) -> Normal:
        mean = self.fc(x)
        std = torch.full_like(mean, self.std)
        return self.normal_cls(mean, std)
