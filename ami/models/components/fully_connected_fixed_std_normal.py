import math

import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.distributions import Normal

SHIFT_ZERO = 1.0 / math.sqrt(2.0 * math.pi)
SCALE_ONE = 1.0 / math.sqrt(2.0)


class FullyConnectedFixedStdNormal(nn.Module):
    """The layer which returns the normal distribution with fixed standard
    deviation.

    https://github.com/MLShukai/ami/issues/117
    """

    def __init__(self, dim_in: int, dim_out: int, std: float = SHIFT_ZERO, normal_cls: type[Normal] = Normal) -> None:
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out)
        self.std = std
        self.normal_cls = normal_cls

    def forward(self, x: Tensor) -> Normal:
        mean = self.fc(x)
        std = torch.full_like(mean, self.std)
        return self.normal_cls(mean, std)


class DeterministicNormal(Normal):
    """Always samples only mean."""

    loc: Tensor

    def sample(self, sample_shape: Size = Size()) -> Tensor:
        return self.rsample(sample_shape)

    def rsample(self, sample_shape: Size = Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        return self.loc.expand(shape)
