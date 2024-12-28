from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.step_data import DataKeys, StepData
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ThreadSafeInferenceWrapper
from ami.utils import Modality

from .base_agent import BaseAgent
from .unimodal_encoding_agent import UnimodalEncodingAgent


class MultimodalTemporalEncodingAgent(BaseAgent[Mapping[Modality, Tensor], Tensor]):
    """Encodes the multimodal observation with temporal information."""

    @override
    def __init__(
        self,
        initial_hidden: Tensor,
        unimodal_agents: Mapping[Modality, UnimodalEncodingAgent | BaseAgent[Tensor, Tensor]] | None = None,
    ) -> None:
        """Initializes the MultimodalTemporalEncodingAgent.

        Args:
            initial_hidden (Tensor): The initial hidden state for the encoder.
            unimodal_agents (Mapping[Modality, UnimodalEncodingAgent | BaseAgent[Tensor, Tensor]] | None, optional):
                A mapping of modalities to their corresponding unimodal encoding agents. Defaults to None.
        """

        if unimodal_agents is None:
            unimodal_agents = {}

        super().__init__(*unimodal_agents.values())
        self.encoder_hidden_state = initial_hidden
        self.unimodal_agents = unimodal_agents

    @override
    def on_inference_models_attached(self) -> None:
        super().on_inference_models_attached()

        self.encoder: ThreadSafeInferenceWrapper[nn.Module] = self.get_inference_model(
            ModelNames.MULTIMODAL_TEMPORAL_ENCODER
        )

    @override
    def on_data_collectors_attached(self) -> None:
        super().on_data_collectors_attached()
        self.collector = self.get_data_collector(BufferNames.MULTIMODAL_TEMPORAL)

    @override
    def step(self, observation: Mapping[Modality, Tensor]) -> Tensor:

        observation = dict(observation)  # copy

        for modality, agent in self.unimodal_agents.items():
            observation[modality] = agent.step(observation[modality])

        self.collector.collect(
            StepData(
                {
                    DataKeys.OBSERVATION: observation,
                    DataKeys.HIDDEN: self.encoder_hidden_state,
                }
            )
        )
        out, self.encoder_hidden_state = self.encoder(observation, self.encoder_hidden_state)
        return out

    @override
    def save_state(self, path: Path) -> None:
        super().save_state(path)
        path.mkdir()
        torch.save(self.encoder_hidden_state, path / "encoder_hidden_state.pt")
        for modality, agent in self.unimodal_agents.items():
            agent.save_state(path / modality)

    @override
    def load_state(self, path: Path) -> None:
        super().load_state(path)
        self.encoder_hidden_state = torch.load(path / "encoder_hidden_state.pt")
        for modality, agent in self.unimodal_agents.items():
            agent.load_state(path / modality)
