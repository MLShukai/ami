from torch import Tensor

from ...data.step_data import DataKeys, StepData
from ...models.model_names import ModelNames
from .base_agent import BaseAgent


class AudioEncodingAgent(BaseAgent[Tensor, Tensor]):
    """Encodes the observed audio with `AUDIO_ENCODER` model."""

    def __init__(self) -> None:
        super().__init__()
        self.step_data = StepData()

    def on_inference_models_attached(self) -> None:
        self.audio_encoder = self.get_inference_model(ModelNames.AUDIO_ENCODER)

    def step(self, observation: Tensor) -> Tensor:
        encoded: Tensor = self.audio_encoder(observation)
        self.step_data[DataKeys.OBSERVATION] = observation
        self.step_data[DataKeys.EMBED_OBSERVATION] = encoded
        self.data_collectors.collect(self.step_data)
        return encoded
