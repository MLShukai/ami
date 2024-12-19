from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from ami.utils import Modality

from .base_agent import BaseAgent
from .curiosity_agent import CuriosityAgent
from .multimodal_temporal_encoding_agent import MultimodalTemporalEncodingAgent


class MultimodalTemporalCuriosityAgent(BaseAgent[Mapping[Modality, Tensor], Tensor]):
    """Agent that combines multimodal encoding, temporal encoding, and
    curiosity-driven exploration."""

    def __init__(
        self,
        multimodal_temporal_agent: MultimodalTemporalEncodingAgent,
        curiosity_agent: CuriosityAgent,
        include_action_modality: bool,
        initial_action: Tensor | None = None,
    ) -> None:
        super().__init__(multimodal_temporal_agent, curiosity_agent)

        self.multimodal_temporal_agent = multimodal_temporal_agent
        self.curiosity_agent = curiosity_agent
        if include_action_modality and initial_action is None:
            raise ValueError("Must provide `initial_action` tensor with including action modality.")

        self._include_action_modality = include_action_modality
        self._previous_action = initial_action

    def step(self, observation: Mapping[Modality, Tensor]) -> Tensor:
        """Process multimodal observation and return action."""

        observation = dict(observation)  # copy

        if self._previous_action is not None:
            observation[Modality.ACTION] = self._previous_action

        encoded = self.multimodal_temporal_agent.step(observation)

        action = self.curiosity_agent.step(encoded)

        if self._include_action_modality:
            self._previous_action = action.clone()

        return action

    def save_state(self, path: Path) -> None:
        super().save_state(path)
        path.mkdir()
        self.multimodal_temporal_agent.save_state(path / "multimodal_temporal")
        self.curiosity_agent.save_state(path / "curiosity")

        torch.save(self._previous_action, path / "previous_action.pt")

    def load_state(self, path: Path) -> None:
        super().load_state(path)
        self.multimodal_temporal_agent.load_state(path / "multimodal_temporal")
        self.curiosity_agent.load_state(path / "curiosity")

        self._previous_action = torch.load(path / "previous_action.pt")
