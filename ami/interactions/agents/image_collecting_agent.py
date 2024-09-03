from torch import Tensor

from ...data.step_data import DataKeys, StepData
from .base_agent import BaseAgent


class ImageCollectingAgent(BaseAgent[Tensor, None]):
    """Collects the observed image."""

    def __init__(self) -> None:
        super().__init__()
        self.step_data = StepData()

    def step(self, observation: Tensor) -> None:
        self.step_data[DataKeys.OBSERVATION] = observation
        self.data_collectors.collect(self.step_data)
        return
