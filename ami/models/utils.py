"""This file contains utility classes."""
import torch.nn as nn

from .base_model import BaseModel, Inference
from .model_wrapper import InferenceWrapper, ModelWrapper


class InferencesDict(dict[str, Inference]):
    """Aggregates inference classes to share models from the training thread to
    the inference thread."""


class ModelsDict(dict[str, BaseModel]):
    """A dictinonary class for aggregating models to be utilized within the
    `hydra` framework."""

    def send_to_default_device(self) -> None:
        """Sends models to the default computing device."""
        for v in self.values():
            v.to_default_device()

    def create_inferences(self) -> InferencesDict:
        """Creates an `InferencesDict` instance from the given models."""
        return InferencesDict({k: v.create_inference() for k, v in self.items() if v.has_inference})


class InferenceWrappersDict(dict[str, InferenceWrapper[nn.Module]]):
    """Aggregates inference wrapper classes to share models from the training
    thread to the inference thread."""


class ModelWrappersDict(dict[str, ModelWrapper[nn.Module]]):
    """A dictinonary class for aggregating model wrappers to be utilized within
    the `hydra` framework."""

    def send_to_default_device(self) -> None:
        """Sends models to the default computing device."""
        for v in self.values():
            v.to_default_device()

    def create_inferences(self) -> InferenceWrappersDict:
        """Creates an `InferenceWrappersDict` instance from the given model
        wrappers."""
        return InferenceWrappersDict({k: v.create_inference() for k, v in self.items() if v.has_inference})
