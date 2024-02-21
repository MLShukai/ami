import torch.nn as nn
from torch import Tensor

from ...data.step_data import DataKeys, StepData
from ...models.model_wrapper import InferenceWrapper
from ...models.utils import ModelNames
from .base_agent import BaseAgent


class ImageEncodingAgent(BaseAgent[Tensor, Tensor]):
    """Encodes the observed image with `IMAGE_ENCODER` model."""

    def on_inference_models_attached(self) -> None:
        self.image_encoder = self.get_inference_model(ModelNames.IMAGE_ENCODER)

    def step(self, observation: Tensor) -> Tensor:
        self.data_collectors.collect(StepData({DataKeys.OBSERVATION: observation}))
        encoded: Tensor = self.image_encoder(observation)
        return encoded
