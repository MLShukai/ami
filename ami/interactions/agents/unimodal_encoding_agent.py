from typing import TypedDict

from torch import Tensor
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.step_data import DataKeys, StepData
from ami.models.model_names import ModelNames
from ami.utils import Modality

from .base_agent import BaseAgent


class UnimodalEncodingAgent(BaseAgent[Tensor, Tensor]):
    """An agent that encodes unimodal (single modality) observations into
    embeddings."""

    class ModalitySetting(TypedDict):
        model_name: ModelNames
        buffer_name: BufferNames

    @override
    def __init__(self, modality: Modality | str) -> None:
        super().__init__()

        match Modality(modality):
            case Modality.IMAGE:
                modality_setting = self.ModalitySetting(
                    model_name=ModelNames.IMAGE_ENCODER, buffer_name=BufferNames.IMAGE
                )
            case Modality.AUDIO:
                modality_setting = self.ModalitySetting(
                    model_name=ModelNames.AUDIO_ENCODER, buffer_name=BufferNames.AUDIO
                )
            case m:
                raise NotImplementedError(f"Modality {m} is not implemented!")

        self._modality_setting = modality_setting

    @override
    def on_inference_models_attached(self) -> None:
        super().on_inference_models_attached()
        self.encoder = self.get_inference_model(self._modality_setting["model_name"])

    @override
    def on_data_collectors_attached(self) -> None:
        super().on_data_collectors_attached()
        self.collector = self.get_data_collector(self._modality_setting["buffer_name"])

    @override
    def step(self, observation: Tensor) -> Tensor:
        encoded: Tensor = self.encoder(observation)
        self.collector.collect(StepData({DataKeys.OBSERVATION: observation, DataKeys.EMBED_OBSERVATION: encoded}))
        return encoded
