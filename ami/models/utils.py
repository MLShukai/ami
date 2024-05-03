"""This file contains utility classes."""
from collections import UserDict
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

import torch
import torch.nn as nn
from typing_extensions import override

from ami.checkpointing import SaveAndLoadStateMixin

from .model_wrapper import ModelWrapper, ThreadSafeInferenceWrapper


class ModelNames(StrEnum):
    """Enumerates the all model names used in ami system."""

    IMAGE_ENCODER = "image_encoder"


class InferenceWrappersDict(UserDict[str, ThreadSafeInferenceWrapper[nn.Module]]):
    """Aggregates inference wrapper classes to share models from the training
    thread to the inference thread."""


class ModelWrappersDict(UserDict[str, ModelWrapper[nn.Module]], SaveAndLoadStateMixin):
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

    @override
    def save_state(self, path: Path) -> None:
        """Saves the model parameters to `path`."""
        path.mkdir()
        for name, wrapper in self.items():
            model_path = path / (name + ".pt")
            torch.save(wrapper.model.state_dict(), model_path)

    @override
    def load_state(self, path: Path) -> None:
        """Loads the model parameters from `path`."""
        for name, wrapper in self.items():
            model_path = path / (name + ".pt")
            wrapper.model.load_state_dict(torch.load(model_path, map_location=wrapper.device))


def count_model_parameters(model: nn.Module) -> tuple[int, int, int]:
    """Counts the number of model parameters.

    Args:
        model: The model that has parameters to count.

    Returns:
        tuple[int, int, int]: total, trainable, frozen parameter counts.
    """
    num_trainable, num_frozen = 0, 0
    for param in model.parameters():
        if param.requires_grad:
            num_trainable += param.numel()
        else:
            num_frozen += param.numel()
    return num_trainable + num_frozen, num_trainable, num_frozen


ModelParameterCountDictType: TypeAlias = dict[str, dict[str, int]]


def create_model_parameter_count_dict(models: ModelWrappersDict) -> ModelParameterCountDictType:
    """Creates the dict object that holds the model parameter count
    infomations.

    Returning dictionary structure is:

        ```
        {
            _all_: {
                total: ...
                trainable: ...
                frozen:
            },
            <model_name>: {
                ...
            }
        }
        ```
    """

    out: ModelParameterCountDictType = {}
    all_total, all_trainable, all_frozen = 0, 0, 0
    for name, wrapper in models.items():
        total, trainable, frozen = count_model_parameters(wrapper.model)
        out[name] = {"total": total, "trainable": trainable, "frozen": frozen}
        all_total += total
        all_trainable += trainable
        all_frozen += frozen

    out["_all_"] = {"total": all_total, "trainable": all_trainable, "frozen": all_frozen}

    return out
