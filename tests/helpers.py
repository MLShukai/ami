"""This file contains helper objects for testing some features."""
import platform
from typing import Self

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from ami.data.buffers.base_data_buffer import BaseDataBuffer
from ami.data.step_data import DataKeys, StepData


def get_gpu_device() -> torch.device | None:
    """Return the available gpu device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps:0")
    else:
        return None


def skip_if_gpu_is_not_available():
    return pytest.mark.skipif(get_gpu_device() is None, reason="GPU devices are not found!")


class ModelMultiplyP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p = nn.Parameter(torch.randn(()))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.p * input


class DataBufferImpl(BaseDataBuffer):
    def init(self) -> None:
        self.obs: list[torch.Tensor] = []

    def add(self, step_data: StepData) -> None:
        self.obs.append(step_data[DataKeys.OBSERVATION])

    def concatenate(self, new_data: Self) -> None:
        self.obs += new_data.obs

    def make_dataset(self) -> TensorDataset:
        return TensorDataset(torch.stack(self.obs))


def skip_if_platform_is_not_linux():
    return pytest.mark.skipif(platform.system() != "Linux", reason="Platform is not linux.")
