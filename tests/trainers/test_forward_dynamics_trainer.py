from functools import partial

import pytest
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.causal_data_buffer import CausalDataBuffer
from ami.data.step_data import DataKeys, StepData
from ami.data.utils import DataCollectorsDict, DataUsersDict
from ami.models.components.forward_dynamics import ForwardDynamics
from ami.models.components.forward_dynamics_wrapper import ForwardDynamicsWrapper
from ami.models.components.sconv import SConv
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.utils import ModelWrappersDict
from ami.trainers.forward_dynamics_trainer import ForwardDynamicsTrainer

BATCH = 4
DEPTH = 8
DIM = 16
DIM_OBS = 16
DIM_ACTION = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1


class TestForwardDynamicsTrainer:
    @pytest.fixture
    def partial_dataloader(self):
        partial_dataloader = partial(DataLoader, batch_size=16, shuffle=True)
        return partial_dataloader

    @pytest.fixture
    def partial_optimizer(self):
        partial_optimizer = partial(Adam, lr=0.001)
        return partial_optimizer

    @pytest.fixture
    def sconv(self):
        sconv = SConv(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT)
        return sconv

    @pytest.fixture
    def forward_dynamics(self, sconv):
        forward_dynamics = ForwardDynamics(sconv, DIM, DIM_OBS, DIM_ACTION)
        return forward_dynamics

    @pytest.fixture
    def trajectory_step_data(self) -> StepData:
        d = StepData()
        d[DataKeys.EMBED_OBSERVATION] = torch.randn(DIM_OBS)
        d[DataKeys.NEXT_EMBED_OBSERVATION] = torch.randn(DIM_OBS)
        d[DataKeys.ACTION] = torch.randn(DIM_ACTION)
        d[DataKeys.HIDDEN] = torch.randn(DEPTH, DIM)
        return d

    @pytest.fixture
    def trajectory_buffer_dict(self, trajectory_step_data: StepData) -> DataCollectorsDict:
        d = DataCollectorsDict.from_data_buffers(
            **{
                BufferNames.FORWARD_DYNAMICS_TRAJECTORY: CausalDataBuffer.reconstructable_init(
                    256,
                    [
                        DataKeys.EMBED_OBSERVATION,
                        DataKeys.NEXT_EMBED_OBSERVATION,
                        DataKeys.ACTION,
                        DataKeys.HIDDEN,
                    ],
                )
            }
        )

        for _ in range(128):
            d.collect(trajectory_step_data)

        return d

    @pytest.fixture
    def forward_dynamics_wrappers_dict(
        self, device: torch.device, forward_dynamics: ForwardDynamics
    ) -> ModelWrappersDict:
        d = ModelWrappersDict(
            {
                ModelNames.FORWARD_DYNAMICS: ForwardDynamicsWrapper(forward_dynamics, device, True),
            }
        )
        d.send_to_default_device()
        return d

    @pytest.fixture
    def trainer(
        self,
        partial_dataloader,
        partial_optimizer,
        forward_dynamics_wrappers_dict: ModelWrappersDict,
        trajectory_buffer_dict: DataCollectorsDict,
        device: torch.device,
    ) -> ForwardDynamicsTrainer:
        trainer = ForwardDynamicsTrainer(partial_dataloader, partial_optimizer, device)
        trainer.attach_model_wrappers_dict(forward_dynamics_wrappers_dict)
        trainer.attach_data_users_dict(trajectory_buffer_dict.get_data_users())
        return trainer

    def test_run(self, trainer: ForwardDynamicsTrainer) -> None:
        trainer.run()

    def test_is_trainable(self, trainer: ForwardDynamicsTrainer) -> None:
        assert trainer.is_trainable() is True
        trainer.trajectory_data_user.clear()
        assert trainer.is_trainable() is False
