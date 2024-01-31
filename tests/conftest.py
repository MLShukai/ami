"""This file contains the pytest fixtures."""
import pytest
import torch

from ami.data.step_data import DataKeys, StepData
from ami.data.utils import DataCollectorsDict, DataUsersDict
from ami.models.model_wrapper import ModelWrapper
from ami.models.utils import InferencesDict, ModelsDict, ModelWrappersDict
from tests.helpers import DataBufferImpl, ModelImpl, ModelMultiplyP, get_gpu_device


@pytest.fixture
def gpu_device() -> torch.device | None:
    """Fixture for retrieving the available gpu device."""
    return get_gpu_device()


@pytest.fixture
def models_dict(gpu_device: torch.device | None) -> ModelsDict:
    if gpu_device is not None:
        device = gpu_device
    else:
        device = "cpu"

    d = ModelsDict(
        {"model1": ModelImpl("cpu", True), "model2": ModelImpl("cpu", False), "model3": ModelImpl(device, True)}
    )

    d.send_to_default_device()
    return d


@pytest.fixture
def model_wrappers_dict(gpu_device: torch.device | None) -> ModelWrappersDict:
    if gpu_device is not None:
        device = gpu_device
    else:
        device = "cpu"

    d = ModelWrappersDict(
        {
            "model1": ModelWrapper(ModelMultiplyP(), "cpu", True),
            "model2": ModelWrapper(ModelMultiplyP(), "cpu", True),
            "model_device": ModelWrapper(ModelMultiplyP(), device, True),
            "model_no_inference": ModelWrapper(ModelMultiplyP(), "cpu", False),
        }
    )
    d.send_to_default_device()
    return d


@pytest.fixture
def step_data() -> StepData:
    d = StepData()
    d[DataKeys.OBSERVATION] = torch.randn(10)
    return d


@pytest.fixture
def data_collectors_dict(step_data: StepData) -> DataCollectorsDict:
    d = DataCollectorsDict.from_data_buffers(
        **{
            "buffer1": DataBufferImpl(),
            "buffer2": DataBufferImpl(),
        }
    )

    d.collect(step_data)
    return d


@pytest.fixture
def data_users_dict(data_collectors_dict: DataCollectorsDict) -> DataUsersDict:
    return data_collectors_dict.get_data_users()
