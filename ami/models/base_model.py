from __future__ import annotations

import copy
import threading
from typing import Any, Self

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base deep neural network class for both inference and training threads.

    Override :meth:`convert_to_inference_model` method to use a custom inference class.

    To create the inference model, use the `create_inference` method.
    The device on which the model is running can be accessed via the `device` attribute.
    """

    def __init__(
        self, *args: Any, default_device: torch.device = torch.device("cpu"), has_inference: bool = True, **kwargs: Any
    ) -> None:
        """Constructs the model.

        Args:
            default_device: The default device to be used when the model is created.
                It is used to determine the `device` property if this model has no parameters or buffers.
            has_inference: Indicates whether the model includes an inference component.
        """
        super().__init__(*args, **kwargs)

        self._default_device = default_device
        self.has_inference = has_inference

    @property
    def device(self) -> torch.device:
        """The device on which model is running."""
        for param in self.parameters():
            return param.device
        for buf in self.buffers():
            return buf.device
        return self._default_device

    def to_default_device(self) -> None:
        """Sends the model to the default computing device."""
        self.to(self._default_device)

    def convert_to_inference_model(self, copied_model: Self) -> Inference:
        """Converts the copied model into the inference model.

        Args:
            copied_model: The deep-copied model for the inference thread.
        """
        return Inference(copied_model)

    def create_inference(self) -> Inference:
        """Creates the inference model for the inference thread.

        Do not call when the model has no inference model.
        To prevent erratic outputs due to parameter updates during inference, a model completely copied with its parameters is passed.

        Returns:
            inference: The model used for the inference thread.

        Raises:
            RuntimeError: When `has_inference` is set to False.
        """
        if self.has_inference:
            return self.convert_to_inference_model(copy.deepcopy(self))
        raise RuntimeError("The model has no inference component!")


class Inference:
    """A wrapper class for performing inference with the model in a thread-safe
    manner.

    Please override :meth:`_infer` method creating the custom inference
    model class.
    """

    def __init__(self, model: BaseModel) -> None:
        """Constructs the inference class.

        Args:
            model: The deep neural network model to be used for inference.
        """
        self._model = model
        self._lock = threading.RLock()

    @property
    def model(self) -> BaseModel:
        """Returns the internal dnn model."""
        return self._model

    @model.setter
    def model(self, m: BaseModel) -> None:
        """Sets the model in a thread-safe manner."""
        with self._lock:
            self._model = m

    def _infer(self, *args: Any, **kwds: Any) -> Any:
        """Performs the inference.

        Tensors in `args` and `kwds` are sent to the computing device.
        """
        device = self._model.device
        new_args, new_kwds = [], {}
        for i in args:
            if isinstance(i, torch.Tensor):
                i = i.to(device)
            new_args.append(i)

        for k, v in kwds.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            new_kwds[k] = v

        return self._model(*new_args, **new_kwds)

    def infer_thread_safely(self, *args: Any, **kwds: Any) -> Any:
        """Performs the inference in a thread-safe manner."""
        with self._lock:
            return self._infer(*args, **kwds)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.infer_thread_safely(*args, **kwds)
