# Ref: https://github.com/facebookresearch/ijepa

import math

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Layer for regularization. During training, probabilistically drop some elements to 0. 
    """

    def __init__(self, drop_prob: float):
        super().__init__()
        assert 1.0 > drop_prob >= 0.0
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """During training, Probabilistically drop some elements to 0.

        Args:
            x (torch.Tensor): Shape is [...]

        Returns:
            torch.Tensor: Shape is same as input.
        """
        if not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = torch.rand(shape, device=x.device) >= self.drop_prob
        output = x.div(1 - self.drop_prob) * random_tensor
        return output
