import math
from typing import Literal

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

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        std: float = SHIFT_ZERO,
        normal_cls: type[Normal] | Literal["Normal", "Deterministic"] = Normal,
        squeeze_feature_dim: bool = False,
    ) -> None:
        super().__init__()
        match normal_cls:
            case "Normal":
                normal_cls = Normal
            case "Deterministic":
                normal_cls = DeterministicNormal
            case str():
                raise ValueError("Normal class must be 'Normal' or 'Deterministic'!")

        if squeeze_feature_dim:
            assert dim_out == 1, "Can not squeeze feature dimension!"
        self.fc = nn.Linear(dim_in, dim_out)
        self.std = std
        self.normal_cls = normal_cls
        self.squeeze_feature_dim = squeeze_feature_dim

    def forward(self, x: Tensor) -> Normal:
        mean: Tensor = self.fc(x)
        if self.squeeze_feature_dim:
            mean = mean.squeeze(-1)
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
