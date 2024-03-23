"""This file contains the pytest fixtures."""
import pytest
import torch

from ami.data.step_data import DataKeys, StepData
from ami.data.utils import DataCollectorsDict, DataUsersDict
from ami.interactions.interaction import Interaction
from ami.models.model_wrapper import ModelWrapper
from ami.models.utils import InferenceWrappersDict, ModelWrappersDict
from ami.threads.base_thread import attach_shared_objects_pool_to_threads
from ami.threads.inference_thread import InferenceThread
from ami.threads.main_thread import MainThread
from ami.threads.training_thread import TrainingThread
from ami.trainers.utils import TrainersList
from tests.helpers import (
    AgentImpl,
    DataBufferImpl,
    EnvironmentImpl,
    ModelMultiplyP,
    TrainerImpl,
    get_gpu_device,
)


@pytest.fixture
def gpu_device() -> torch.device | None:
    """Fixture for retrieving the available gpu device."""
    return get_gpu_device()


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
def inference_wrappers_dict(model_wrappers_dict: ModelWrappersDict) -> InferenceWrappersDict:
    return model_wrappers_dict.inference_wrappers_dict


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


@pytest.fixture
def interaction() -> Interaction:
    return Interaction(EnvironmentImpl(), AgentImpl())


@pytest.fixture
def trainers() -> TrainersList:
    return TrainersList(*[TrainerImpl()])


@pytest.fixture
def thread_objects(
    interaction, data_collectors_dict, trainers, model_wrappers_dict
) -> tuple[MainThread, InferenceThread, TrainingThread]:
    """Instantiates main, inference, training threads and attach shared object
    pool to them."""
    mt = MainThread()
    it = InferenceThread(interaction, data_collectors_dict)
    tt = TrainingThread(trainers, model_wrappers_dict)
    attach_shared_objects_pool_to_threads(mt, it, tt)
    return mt, it, tt


@pytest.fixture
def device(gpu_device: torch.device | None) -> torch.device:
    if gpu_device is None:
        return torch.device("cpu")
    return gpu_device
