import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution


class PolicyOrValueNetwork(nn.Module):
    """Module for policy or value network."""

    def __init__(
        self,
        observation_projection: nn.Module,
        forward_dynamics_hidden_projection: nn.Module,
        observation_hidden_projection: nn.Module,
        core_model: nn.Module,
        dist_head: nn.Module,
    ) -> None:
        """Constructs the model with components.

        Args:
            observation_projection: Layer that processes observations only.
            forward_dynamics_hidden_projection: Layer that processes hidden states of the Forward Dynamics model only.
            observation_hidden_projection: Layer that receives and integrates observations and hidden states.
            core_model: Layer that processes the integrated tensor.
            dist_head: Layer that generates prediction distribution.
        """
        super().__init__()
        self.observation_projection = observation_projection
        self.forward_dynamics_hidden_projection = forward_dynamics_hidden_projection
        self.observation_hidden_projection = observation_hidden_projection
        self.core_model = core_model
        self.dist_head = dist_head

    def forward(self, observation: Tensor, forward_dynamics_hidden: Tensor) -> Distribution:
        """Returns the prediction distribution."""
        obs_embed = self.observation_projection(observation)
        hidden_embed = self.forward_dynamics_hidden_projection(forward_dynamics_hidden)
        x = self.observation_hidden_projection(obs_embed, hidden_embed)
        h = self.core_model(x)
        return self.dist_head(h)
