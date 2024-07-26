import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal


class FullyConnectedNormal(nn.Module):
    """The layer which returns the normal distribution."""

    def __init__(self, dim_in: int, dim_out: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.fc_mean = nn.Linear(dim_in, dim_out)
        self.fc_std = nn.Linear(dim_in, dim_out)
        self.eps = eps

    def forward(self, x: Tensor) -> Normal:
        mean = self.fc_mean(x)
        std = nn.functional.softplus(self.fc_std(x)) + self.eps
        return Normal(mean, std)
