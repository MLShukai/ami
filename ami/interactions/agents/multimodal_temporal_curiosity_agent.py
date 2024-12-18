from collections.abc import Mapping
from pathlib import Path

from torch import Tensor

from ami.utils import Modality

from .base_agent import BaseAgent
from .curiosity_agent import CuriosityAgent
from .multimodal_temporal_encoding_agent import MultimodalTemporalEncodingAgent
from .unimodal_encoding_agent import UnimodalEncodingAgent


class MultimodalTemporalCuriosityAgent(BaseAgent[Mapping[Modality, Tensor], Tensor]):
    """Agent that combines multimodal encoding, temporal encoding, and
    curiosity-driven exploration.

    This agent processes multimodal observations through:
    1. Unimodal encoding for each modality
    2. Temporal encoding of combined multimodal features
    3. Curiosity-driven action selection
    """

    def __init__(
        self,
        unimodal_agents: Mapping[Modality, UnimodalEncodingAgent],
        temporal_agent: MultimodalTemporalEncodingAgent,
        curiosity_agent: CuriosityAgent,
        include_action_modality: bool,
        initial_action: Tensor | None = None,
    ) -> None:
        super().__init__(*unimodal_agents.values(), temporal_agent, curiosity_agent)

        self.unimodal_agents = unimodal_agents
        self.temporal_agent = temporal_agent
        self.curiosity_agent = curiosity_agent
        if include_action_modality and initial_action is None:
            raise ValueError("Must provide `initial_action` tensor with including action modality.")

        self._include_action_modality = include_action_modality
        self._previous_action = initial_action

    def step(self, observation: Mapping[Modality, Tensor]) -> Tensor:
        """Process multimodal observation and return action."""

        # Step 1: Unimodal encoding
        encoded_obs = {}
        for modality, agent in self.unimodal_agents.items():
            encoded_obs[modality] = agent.step(observation[modality])
        if self._previous_action is not None:
            encoded_obs[Modality.ACTION] = self._previous_action

        # Step 2: Temporal encoding
        temporal_encoded = self.temporal_agent.step(encoded_obs)

        # Step 3: Curiosity-driven action selection
        action = self.curiosity_agent.step(temporal_encoded)

        if self._include_action_modality:
            self._previous_action = action.clone()

        return action

    def save_state(self, path: Path) -> None:
        super().save_state(path)
        path.mkdir()
        for modality, agent in self.unimodal_agents.items():
            agent.save_state(path / modality)
        self.temporal_agent.save_state(path / "temporal")
        self.curiosity_agent.save_state(path / "curiosity")

    def load_state(self, path: Path) -> None:
        super().load_state(path)
        for modality, agent in self.unimodal_agents.items():
            agent.load_state(path / modality)
        self.temporal_agent.load_state(path / "temporal")
        self.curiosity_agent.load_state(path / "curiosity")
