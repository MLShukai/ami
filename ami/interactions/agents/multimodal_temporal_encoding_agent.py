from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.step_data import DataKeys, StepData
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ThreadSafeInferenceWrapper
from ami.models.temporal_encoder import MultimodalTemporalEncoder
from ami.utils import Modality

from .base_agent import BaseAgent


class MultimodalTemporalEncodingAgent(BaseAgent[Mapping[Modality, Tensor], Tensor]):
    """Encodes the multimodal observation with temporal information."""

    @override
    def __init__(
        self,
        initial_hidden: Tensor,
    ) -> None:
        super().__init__()
        self.encoder_hidden_state = initial_hidden

    @override
    def on_inference_models_attached(self) -> None:
        super().on_inference_models_attached()

        self.encoder: ThreadSafeInferenceWrapper[MultimodalTemporalEncoder] = self.get_inference_model(
            ModelNames.MULTIMODAL_TEMPORAL_ENCODER
        )

    @override
    def on_data_collectors_attached(self) -> None:
        super().on_data_collectors_attached()
        self.collector = self.get_data_collector(BufferNames.MULTIMODAL_TEMPORAL)

    @override
    def step(self, observation: Mapping[Modality, Tensor]) -> Tensor:
        self.collector.collect(
            StepData(
                {
                    DataKeys.OBSERVATION: observation,
                    DataKeys.HIDDEN: self.encoder_hidden_state,
                }
            )
        )
        out, self.encoder_hidden_state, _ = self.encoder(observation, self.encoder_hidden_state)
        return out

    @override
    def save_state(self, path: Path) -> None:
        super().save_state(path)
        path.mkdir()
        torch.save(self.encoder_hidden_state, path / "encoder_hidden_state.pt")

    @override
    def load_state(self, path: Path) -> None:
        super().load_state(path)
        self.encoder_hidden_state = torch.load(path / "encoder_hidden_state.pt")
