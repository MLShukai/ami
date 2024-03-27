from torch import Tensor
from torch.distributions.normal import Normal

from ..model_wrapper import ModelWrapper
from .forward_dynamics import ForwardDynamics


class ForwardDynamicsWrapper(ModelWrapper[ForwardDynamics]):
    # (batch, len, dim), (batch, depth, dim) -> Normal(batch, len, dim), (batch, depth, len, dim) or
    # (len, dim), (depth, dim) -> Normal(len, dim), (depth, len, dim)
    def forward(self, obs: Tensor, action: Tensor, hidden_stack: Tensor) -> tuple[Normal, Tensor]:
        is_no_batch = len(obs.shape) == 2
        if is_no_batch:
            obs = obs.unsqueeze(0)  # (batch, len, dim_obs)
            action = action.unsqueeze(0)  # (batch, len, dim_obs)
            hidden_stack = hidden_stack.unsqueeze(0)  # (batch, depth, dim_hidden)

        # (batch, len, dim_obs), (batch, len, dim_obs), (batch, depth, len, dim_hidden),
        obs_hat_mean, obs_hat_std, hidden_out = self.model(obs, action, hidden_stack)

        if is_no_batch:
            obs_hat_mean = obs_hat_mean.squeeze(0)  # (len, dim_obs)
            obs_hat_std = obs_hat_std.squeeze(0)  # (len, dim_obs)
            hidden_out = hidden_out.squeeze(0)  # (depth, len, dim_hidden)
        obs_hat_dist = Normal(obs_hat_mean, obs_hat_std)  # (len, dim_obs)
        return obs_hat_dist, hidden_out

    # (batch, dim), (batch, depth, dim) -> Normal(batch, dim), (batch, depth, dim) or
    # (dim), (dim) -> Normal(dim), (depth, dim)
    def infer(self, obs: Tensor, action: Tensor, hidden_stack: Tensor) -> tuple[Normal, Tensor]:
        is_no_batch = len(obs.shape) == 1
        if is_no_batch:
            obs = obs.unsqueeze(0)  # (batch, dim_obs)
            action = action.unsqueeze(0)  # (batch, dim_action)
            hidden_stack = hidden_stack.unsqueeze(0)  # (batch, depth, dim_hidden)
        obs = obs.unsqueeze(1)  # (batch, len, dim_obs)
        action = action.unsqueeze(1)  # (batch, len, dim_action)

        # (batch, len, dim_obs), (batch, len, dim_obs), (batch, depth, len, dim_hidden),
        obs_hat_mean, obs_hat_std, hidden_out = self.model(obs, action, hidden_stack)

        obs_hat_mean = obs_hat_mean.squeeze(1)  # (batch, dim_obs)
        obs_hat_std = obs_hat_std.squeeze(1)  # (batch, dim_obs)
        hidden_out = hidden_out.squeeze(2)  # (batch, depth, dim)
        if is_no_batch:
            obs_hat_mean = obs_hat_mean.squeeze(0)  # (dim_obs)
            obs_hat_std = obs_hat_std.squeeze(0)  # (dim_obs)
            hidden_out = hidden_out.squeeze(0)  # (depth, dim)
        obs_hat_dist = Normal(obs_hat_mean, obs_hat_std)
        return obs_hat_dist, hidden_out
