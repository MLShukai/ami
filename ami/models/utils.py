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
    """A dictionary class for aggregating model wrappers to be utilized within
    the `hydra` framework.

    NOTE:
        The `inference_wrappers_dict` attribute (`property`) is a singleton for this class.
        There is always only one instance of `InferenceWrappersDict` corresponding to this `ModelWrappersDict` class.
    """

    _inference_wrappers_dict: InferenceWrappersDict | None = None

    def send_to_default_device(self) -> None:
        """Sends models to the default computing device."""
        for v in self.values():
            v.to_default_device()

    def _create_inferences(self) -> InferenceWrappersDict:
        """Creates an instance of `InferenceWrappersDict` from the given model
        wrappers."""
        return InferenceWrappersDict({k: v.create_inference() for k, v in self.items() if v.has_inference})

    @property
    def inference_wrappers_dict(self) -> InferenceWrappersDict:
        """Returns the corresponding `InferenceWrappersDict` for this class.

        NOTE: Returns a reference to the existing instance if it has already been created.
        """
        if self._inference_wrappers_dict is None:
            self._inference_wrappers_dict = self._create_inferences()
            return self._inference_wrappers_dict
        else:
            return self._inference_wrappers_dict
