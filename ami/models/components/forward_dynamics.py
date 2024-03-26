import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal

from .stacked_hidden_state import StackedHiddenState


class ForwardDynamics(nn.Module):
    def __init__(
        self,
        stacked_hidden_state: StackedHiddenState,
        dim: int,
        dim_obs: int,
        dim_action: int,
    ):
        super().__init__()
        self.stacked_hidden_state = stacked_hidden_state
        self.fc_in = nn.Linear(dim_obs + dim_action, dim)
        self.fc_out_mean = nn.Linear(dim, dim_obs)
        self.fc_out_std = nn.Linear(dim, dim_obs)

    def forward(self, obs: Tensor, action: Tensor, hidden_stack: Tensor) -> tuple[Normal, Tensor]:
        x = self.fc_in(torch.cat((obs, action), dim=-1))
        x, hidden_out = self.stacked_hidden_state(x, hidden_stack)
        obs_hat_mean = self.fc_out_mean(x)
        obs_hat_std = torch.nn.functional.softplus(self.fc_out_std(x)) + 1e-7
        obs_dist = Normal(obs_hat_mean, obs_hat_std)
        return obs_dist, hidden_out
