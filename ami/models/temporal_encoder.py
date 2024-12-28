from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution

from ami.utils import Modality

from .components.stacked_hidden_state import StackedHiddenState
from .model_wrapper import ModelWrapper


class MultimodalTemporalEncoder(nn.Module):
    """Multimodal temporal encoder framework module."""

    def __init__(
        self,
        observation_flattens: Mapping[str, nn.Module],
        flattened_obses_projection: nn.Module,
        core_model: StackedHiddenState,
        obs_hat_dist_heads: Mapping[str, nn.Module],
    ) -> None:
        """Initializes MultimodalTemporalEncoder.

        Args:
            observation_flattens (Mapping[str, nn.Module]): The dict contains the flatten module for each modality
            flattened_obses_projection (nn.Module): Projects the concatenated flattened observations to fixed size vector.
            core_model (StackedHiddenState): Core model for temporal encoding.
            obs_hat_dist_heads (Mapping[str, nn.Module]): Observation prediction heads for each modality

        Raises:
            KeyError: If keys between observation_flattens and obs_hat_dist_heads are not matched.
        """
        super().__init__()
        if obs_hat_dist_heads.keys() != observation_flattens.keys():
            raise KeyError(
                "The keys between observation_flattens and obs_hat_dist_heads are not matched! "
                f"observation_flattens keys: {observation_flattens.keys()}, "
                f"obs_hat_dist_heads keys: {obs_hat_dist_heads.keys()}"
            )

        for modality in obs_hat_dist_heads.keys():
            Modality(modality)  # Simple error check

        self.observation_flattens = nn.ModuleDict(observation_flattens)

        self.flattened_obses_projection = flattened_obses_projection
        self.core_model = core_model
        self.obs_hat_dist_heads = nn.ModuleDict(obs_hat_dist_heads)

    def forward(
        self, observations: Mapping[str, Tensor], hidden: Tensor
    ) -> tuple[Tensor, Tensor, Mapping[str, Distribution]]:
        """Forward path of multimodal temporal encoder.

        Args:
            observations (Mapping[str, Tensor]): The dict that contains the tensors
            hidden (Tensor): Hidden state tensor for temporal module.

        Returns:
            tuple[Tensor, Tensor, Mapping[str, Distribution]]: embed_obs, next_hidden, predicted_observations.
        """
        if observations.keys() != self.observation_flattens.keys():
            raise KeyError("Observations keys are not matched!")

        obs_flats = [layer(observations[k]) for k, layer in self.observation_flattens.items()]

        x = self.flattened_obses_projection(torch.cat(obs_flats, dim=-1))
        x, next_hidden = self.core_model(x, hidden)
        obs_hat_dists = {k: layer(x) for k, layer in self.obs_hat_dist_heads.items()}
        return x, next_hidden, obs_hat_dists


def inference_forward(
    wrapper: ModelWrapper[MultimodalTemporalEncoder], observation: Mapping[str, Tensor], hidden: Tensor
) -> tuple[Tensor, Tensor]:
    """Custom encoder infer procedure for multimodal temporal encoder.

    Args:
        wrapper (ModelWrapper[MultimodalTemporalEncoder]): Wrapper instance
        observation (Mapping[str, Tensor]): observation to encoded.
        hidden (Tensor): hidden state for temporal encoding

    Returns:
        tuple[Tensor, Tensor]: (encoded, next_hidden).
    """
    device = wrapper.device
    observation = {k: v.to(device) for k, v in observation.items()}
    x, next_hidden, _ = wrapper(observation, hidden.to(device))
    return x, next_hidden
