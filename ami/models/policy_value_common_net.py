from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Distribution


class PolicyValueCommonNet(nn.Module):
    """Modules with shared models for policy and value functions."""

    @abstractmethod
    def forward(self, *args: Any, **kwds: Any) -> tuple[Distribution, torch.Tensor]:
        """Returns the action distribution and estimated value."""
        raise NotImplementedError
