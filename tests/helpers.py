"""This file contains helper objects for testing some features."""
import platform
from typing import Self

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from ami.data.buffers.base_data_buffer import BaseDataBuffer
from ami.data.step_data import DataKeys, StepData
from ami.interactions.agents.base_agent import BaseAgent
from ami.interactions.environments.base_environment import BaseEnvironment
from ami.models.model_wrapper import ModelWrapper
from ami.trainers.base_trainer import BaseTrainer


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


class TrainerImpl(BaseTrainer):
    def on_model_wrappers_dict_attached(self) -> None:
        super().on_model_wrappers_dict_attached()

        self.model1: ModelWrapper[ModelMultiplyP] = self.get_training_model("model1")
        self.model2: ModelWrapper[ModelMultiplyP] = self.get_frozen_model("model2")

    def on_data_users_dict_attached(self) -> None:
        super().on_data_users_dict_attached()

        self.data_user = self.get_data_user("buffer1")

    def train(self) -> None:
        dataset = self.data_user.get_dataset()
        data = dataset[0][0]
        self.model1(data)
        self.model2(data)


class AgentImpl(BaseAgent[str, str]):
    def on_inference_models_attached(self) -> None:
        self.model1 = self.get_inference_model("model1")

    def on_data_collectors_attached(self) -> None:
        self.data_collector1 = self.get_data_collector("buffer1")

    def step(self, observation: str) -> str:
        return "action"


class EnvironmentImpl(BaseEnvironment[str, str]):
    def observe(self) -> str:
        return "observation"

    def affect(self, action: str) -> None:
        pass
