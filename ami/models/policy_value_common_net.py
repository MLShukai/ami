import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution


class PolicyValueCommonNet(nn.Module):
    """Module with shared models for policy and value functions."""

    def __init__(
        self,
        observation_projection: nn.Module,
        forward_dynamics_hidden_projection: nn.Module,
        observation_hidden_projection: nn.Module,
        core_model: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
    ) -> None:
        """Constructs the model with components.

        Args:
            observation_projection: Layer that processes observations only.
            forward_dynamics_hidden_projection: Layer that processes hidden states of the Forward Dynamics model only.
            observation_hidden_projection: Layer that receives and integrates observations and hidden states.
            core_model: Layer that processes the integrated tensor.
            policy_head: Layer that predicts actions.
            value_head: Layer that predicts state values.
        """
        super().__init__()
        self.observation_projection = observation_projection
        self.forward_dynamics_hidden_projection = forward_dynamics_hidden_projection
        self.observation_hidden_projection = observation_hidden_projection
        self.core_model = core_model
        self.policy_head = policy_head
        self.value_head = value_head

    def forward(self, observation: Tensor, forward_dynamics_hidden: Tensor) -> tuple[Distribution, Tensor]:
        """Returns the action distribution and estimated value."""
        obs_embed = self.observation_projection(observation)
        hidden_embed = self.forward_dynamics_hidden_projection(forward_dynamics_hidden)
        x = self.observation_hidden_projection(obs_embed, hidden_embed)
        h = self.core_model(x)
        return self.policy_head(h), self.value_head(h)


class SelectObservation(nn.Module):
    """Selects the observation only to keep compatibility with
    `PritmiveAMI`."""

    def forward(self, observation: Tensor, hidden: Tensor) -> Tensor:
        return observation


class ConcatFlattenedObservationAndStackedHidden(nn.Module):
    """Concatenates the flattened observation and stacked hidden states.

    Shape:
        - flattened_obs: (*, N)
        - stacked_hidden: (*, D, N)

        Return shape: (*, D+1, N) if transpose is False, else (*, N, D+1)
    """

    def __init__(self, transpose: bool = False) -> None:
        super().__init__()
        self.transpose = transpose

    def forward(self, flattened_obs: Tensor, stacked_hidden: Tensor) -> Tensor:
        out = torch.cat([flattened_obs.unsqueeze(-2), stacked_hidden], dim=-2)
        if self.transpose:
            out = out.transpose(-2, -1)
        return out


class LerpStackedHidden(nn.Module):
    """Linear interpolation along depth of stacked hidden.

    Shape:
        - stacked_hidden: (*, D, N)

        Return shape: (*, N)
    """

    def __init__(self, dim: int, depth: int, num_head: int) -> None:
        super().__init__()
        self.hidden_proj = nn.Parameter(torch.randn(depth, dim, dim) * (dim**-0.5))
        self.logit_coef_proj = nn.Linear(depth * dim, depth)
        self.num_head = num_head
        self.norm = nn.GroupNorm(num_head, num_head)

    def forward(self, stacked_hidden: Tensor) -> Tensor:
        is_batch = len(stacked_hidden.shape) == 3
        if not is_batch:
            stacked_hidden = stacked_hidden.unsqueeze(0)

        batch, depth, dim = stacked_hidden.shape
        stacked_hidden = self.norm(stacked_hidden.reshape(batch * depth, self.num_head, dim // self.num_head)).reshape(
            batch, depth, dim
        )

        logit_coef = self.logit_coef_proj(stacked_hidden.reshape(batch, depth * dim))

        out = torch.einsum(
            "bd,dij,bdj->bi",
            nn.functional.softmax(logit_coef, dim=-1).squeeze(-1),
            self.hidden_proj,
            stacked_hidden,
        )
        if not is_batch:
            out = out.squeeze(0)
        return out


class ConcatFlattenedObservationAndLerpedHidden(nn.Module):
    """Concatenates the flattened observation and stacked hidden states.

    Shape:
        - flattened_obs: (*, N_OBS)
        - lerped_hidden: (*, N_HIDDEN)

        Return shape: (*, N_OUT)
    """

    def __init__(self, dim_obs: int, dim_hidden: int, dim_out: int):
        super().__init__()
        self.fc = nn.Linear(dim_obs + dim_hidden, dim_out)

    def forward(self, flattened_obs: Tensor, lerped_hidden: Tensor) -> Tensor:
        return self.fc(torch.cat([flattened_obs, lerped_hidden], dim=-1))
