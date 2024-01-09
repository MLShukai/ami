"""This file contains aggregation classes used for sharing model between the
training and inference threads."""
from .base_model import BaseModel, Inference


class InferencesAggregation(dict[str, Inference]):
    """Aggregates inference classes to share models from the training thread to
    the inference thread."""


class ModelsAggregation(dict[str, BaseModel]):
    """A class for aggregating models to be utilized within the `hydra`
    framework."""

    def send_to_default_device(self) -> None:
        """Sends models to the default computing device."""
        for v in self.values():
            v.to_default_device()

    def create_inferences(self) -> InferencesAggregation:
        """Creates an `InferencesAggregation` instance from the given
        models."""
        return InferencesAggregation({k: v.create_inference() for k, v in self.items() if v.has_inference})
