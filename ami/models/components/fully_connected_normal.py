import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal


class FullyConnectedNormal(nn.Module):
    """The layer which returns the normal distribution."""

    def __init__(self, dim_in: int, dim_out: int, eps: float = 1e-6, squeeze_feature_dim: bool = False) -> None:
        super().__init__()
        if squeeze_feature_dim:
            assert dim_out == 1, "Can not squeeze feature dimension!"

        self.fc_mean = nn.Linear(dim_in, dim_out)
        self.fc_logvar = nn.Linear(dim_in, dim_out)
        self.eps = eps
        self.squeeze_feature_dim = squeeze_feature_dim

        nn.init.normal_(self.fc_logvar.weight, 0, 0.01)
        nn.init.constant_(self.fc_logvar.bias, 0)

    def forward(self, x: Tensor) -> Normal:
        mean: Tensor = self.fc_mean(x)
        std: Tensor = torch.exp(0.5 * self.fc_logvar(x)) + self.eps
        if self.squeeze_feature_dim:
            mean = mean.squeeze(-1)
            std = std.squeeze(-1)
        return Normal(mean, std)
