from __future__ import annotations

import copy
import threading
from typing import Any, Generic, TypeVar

import torch
import torch.nn as nn

ModuleType = TypeVar("ModuleType", bound=nn.Module)


class ModelWrapper(nn.Module, Generic[ModuleType]):
    """Wraps the deep neural network model (`nn.Module`) for the AMI system.

    The `forward` method is utilized during the training procedure (in the training thread),
    while the `infer` method is used during the inference procedure (in the inference thread).

    Override the `infer` method to implement a custom inference flow.

    The device on which the model is running can be determined through the `device` attribute.
    """

    def __init__(
        self,
        model: ModuleType,
        default_device: torch.device = torch.device("cpu"),
        has_inference: bool = True,
    ) -> None:
        """Constructs the model wrapper.

        Args:
            model: The deep neural network model.
            default_device: The default device for model creation. This is used to determine the `device` property if the model lacks parameters or buffers.
            has_inference: Specifies whether the wrapper contains an inference component.
        """

        super().__init__()
        self.model = model
        self._default_device = torch.device(default_device)
        self.has_inference = has_inference

    def forward(self, *args: Any, **kwds: Any) -> Any:
        """Executes the forward path for the training."""
        return self.model(*args, **kwds)

    @property
    def device(self) -> torch.device:
        """The device on which model is running."""
        for param in self.model.parameters():
            return param.device
        for buf in self.model.buffers():
            return buf.device
        return self._default_device

    def infer(self, *args: Any, **kwds: Any) -> Any:
        """Performs the inference.

        Tensors in `args` and `kwds` are sent to the computing device.
        If you override this method, be careful to send the input tensor
        to the computing device.
        """
        device = self.device
        new_args, new_kwds = [], {}
        for i in args:
            if isinstance(i, torch.Tensor):
                i = i.to(device)
            new_args.append(i)

        for k, v in kwds.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            new_kwds[k] = v

        return self.model(*new_args, **new_kwds)

    def to_default_device(self) -> None:
        """Sends the model to the default computing device."""
        self.model.to(self._default_device)

    def freeze_model(self) -> None:
        """Freezes the model parameters (sets `requires_grad` to `False`) and
        switches to evaluation mode."""
        for p in self.model.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze_model(self) -> None:
        """Un-freezes the model parameters (sets `requires_grad` to `True`) and
        switches to training mode."""
        for p in self.model.parameters():
            p.requires_grad = True
        self.train()

    def create_inference(self) -> ThreadSafeInferenceWrapper[ModuleType]:
        """Creates the inference wrapper for the inference thread.

        Do not call when the model wrapper has no inference model.
        To prevent erratic outputs due to parameter updates during inference, a model wrapper completely copied with its parameters is passed.

        Returns:
            inference_wrapper: The wrapper used for the inference thread.

        Raises:
            RuntimeError: When `has_inference` is set to False.
        """
        if self.has_inference:
            copied_self = copy.deepcopy(self)
            copied_self.freeze_model()
            return ThreadSafeInferenceWrapper(copied_self)

        raise RuntimeError("The model has no inference component!")


class ThreadSafeInferenceWrapper(Generic[ModuleType]):
    """A wrapper class for performing inference with the model in a thread-safe
    manner.

    This class is intended for use in the inference thread, as part of
    the agent component.
    """

    def __init__(self, wrapper: ModelWrapper[ModuleType]) -> None:
        """Constructs the class.

        Args:
            wrapper: An instance of the ModelWrapper class.
        """
        self._wrapper = wrapper
        self._lock = threading.RLock()

    @property
    def model(self) -> ModuleType:
        """Returns the internal dnn model.

        Do not access this property in the inference thread. This
        property is used to switch the model between training and
        inference model."
        """
        return self._wrapper.model

    @model.setter
    def model(self, m: ModuleType) -> None:
        """Sets the model in a thread-safe manner."""
        with self._lock:
            self._wrapper.model = m

    @torch.inference_mode()
    def infer(self, *args: Any, **kwds: Any) -> Any:
        """Performs the inference in a thread-safe manner."""
        with self._lock:
            return self._wrapper.infer(*args, **kwds)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.infer(*args, **kwds)
