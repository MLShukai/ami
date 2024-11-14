import torch
import torch.nn as nn
from torch import Tensor

from .components.fully_connected_fixed_std_normal import (
    DeterministicNormal,
    FullyConnectedFixedStdNormal,
)


class ContinousActionPredictor(nn.Module):
    """The action predicting module which implemented in PrimitiveAMI.

    See: https://github.com/MLShukai/PrimitiveAMI/blob/main/src/models/components/action_predictor/continuous_action_predictor.py
    """

    def __init__(self, dim_embed: int, dim_action: int, dim_hidden: int | None = None) -> None:
        """
        Args:
            dim_embed: Number of dimensions of embedded observation.
            dim_action: Number of dimensions of action.
            dim_hidden: Number of dimensions of tensors in hidden layer.
        """
        super().__init__()
        if dim_hidden is None:
            dim_hidden = dim_embed  # 2 * dim_embed -> dim_embed

        self.fc_in = nn.Linear(dim_embed * 2, dim_hidden)
        self.fc_hidden = nn.Linear(dim_hidden, dim_hidden)
        self.fc_out = FullyConnectedFixedStdNormal(dim_hidden, dim_action, normal_cls=DeterministicNormal)

    def forward(self, embed_obs: Tensor, next_embed_obs: Tensor) -> DeterministicNormal:
        x = torch.concat([embed_obs, next_embed_obs], dim=-1)
        x = self.fc_in(x)
        x = torch.relu(x)
        x = self.fc_hidden(x)
        x = torch.relu(x)
        out = self.fc_out(x)
        return out
