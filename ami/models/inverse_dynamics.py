import torch.nn as nn
from torch import Tensor


class InverseDynamics(nn.Module):
    """Inverse Dynamics framework module.

    This model predicts the action `a_t` from observations `obs` and
    `next_obs`.
    """

    def __init__(
        self,
        observation_encoder: nn.Module,
        action_predictor: nn.Module,
    ) -> None:
        """
        Args:
            observation_encoder: The encoder module to process observations.
            action_predictor: The module to predict actions from encoded observations.
        """
        super().__init__()
        self.observation_encoder = observation_encoder
        self.action_predictor = action_predictor

    def forward(self, obs: Tensor, next_obs: Tensor) -> Tensor:
        embed = self.observation_encoder(obs)
        next_embed = self.observation_encoder(next_obs)
        action_hat = self.action_predictor(embed, next_embed)
        return action_hat
