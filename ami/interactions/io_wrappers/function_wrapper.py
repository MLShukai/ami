from typing import Callable, Generic, TypeVar

import torch

from ami.utils import Modality

from .base_io_wrapper import BaseIOWrapper, WrappedType, WrappingType


class FunctionIOWrapper(BaseIOWrapper[WrappingType, WrappedType], Generic[WrappingType, WrappedType]):
    """The wrapper that allows to use any function as an IO wrapper."""

    def __init__(self, wrap_function: Callable[[WrappingType], WrappedType]):
        self.wrap_function = wrap_function

    def wrap(self, input: WrappingType) -> WrappedType:
        return self.wrap_function(input)


def normalize_tensor(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalizes the input tensor to have a mean of 0 and standard deviation
    of 1.

    Args:
        x (torch.Tensor): Input tensor to be normalized
        epsilon (float): Small value to prevent division by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = torch.mean(x)
    std = torch.std(x)
    return (x - mean) / (std + eps)


T = TypeVar("T")


def to_multimodal_dict(x: T, modality: Modality) -> dict[Modality, T]:
    """Convert unimodal (single tensor) input to multimodal dict."""
    return {Modality(modality): x}
