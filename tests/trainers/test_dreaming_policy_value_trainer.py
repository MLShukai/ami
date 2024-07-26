from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ami.data.step_data import DataKeys, StepData
from ami.data.utils import DataCollectorsDict
from ami.models.components.fully_connected_normal import FullyConnectedNormal
from ami.models.components.sioconv import SioConv
from ami.models.policy_value_common_net import (
    ConcatFlattenedObservationAndLerpedHidden,
    LerpStackedHidden,
)
from ami.models.utils import ModelWrappersDict
from ami.trainers.dreaming_policy_value_trainer import (
    BufferNames,
    DreamingPolicyValueTrainer,
    ForwardDynamcisWithActionReward,
    ModelNames,
    ModelWrapper,
    PolicyOrValueNetwork,
    RandomDataBuffer,
    StepIntervalLogger,
)

BATCH = 4
DEPTH = 8
DIM = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1
DIM_OBS = 32
DIM_ACTION = 8
CHUNK_SIZE = 16
NUM_HEAD = 4


class TestDreamingPolicyValueTrainer:
    @pytest.fixture
    def forward_dynamics(self):
        return ForwardDynamcisWithActionReward(
            nn.Identity(),
            nn.Identity(),
            nn.Linear(DIM_OBS + DIM_ACTION, DIM),
            SioConv(DEPTH, DIM, NUM_HEAD, DIM_FF_HIDDEN, DROPOUT, CHUNK_SIZE),
            FullyConnectedNormal(DIM, DIM_OBS),
            FullyConnectedNormal(DIM, DIM_ACTION),
            FullyConnectedNormal(DIM, 1),
        )

    @pytest.fixture
    def policy_net(self):
        return PolicyOrValueNetwork(
            nn.Identity(),
            LerpStackedHidden(DIM, DEPTH, NUM_HEAD),
            ConcatFlattenedObservationAndLerpedHidden(DIM_OBS, DIM, DIM),
            nn.ReLU(),
            FullyConnectedNormal(DIM, DIM_ACTION),
        )

    @pytest.fixture
    def value_net(self):
        return PolicyOrValueNetwork(
            nn.Identity(),
            LerpStackedHidden(DIM, DEPTH, NUM_HEAD),
            ConcatFlattenedObservationAndLerpedHidden(DIM_OBS, DIM, DIM),
            nn.ReLU(),
            FullyConnectedNormal(DIM, 1),
        )

    @pytest.fixture
    def data_users_dict(self):
        data = StepData()
        data[DataKeys.OBSERVATION] = torch.randn(DIM_OBS)
        data[DataKeys.HIDDEN] = torch.randn(DEPTH, DIM)

        d = DataCollectorsDict.from_data_buffers(
            **{
                BufferNames.DREAMING_INITIAL_STATES: RandomDataBuffer.reconstructable_init(
                    32, [DataKeys.OBSERVATION, DataKeys.HIDDEN]
                )
            }
        )
        d.collect(data)
        d.collect(data)
        d.collect(data)
        d.collect(data)
        return d.get_data_users()

    @pytest.fixture
    def partial_dataloader(self):
        partial_dataloader = partial(DataLoader, batch_size=2, shuffle=True)
        return partial_dataloader

    @pytest.fixture
    def partial_policy_optimizer(self):
        partial_optimizer = partial(Adam, lr=0.001)
        return partial_optimizer

    @pytest.fixture
    def partial_value_optimizer(self):
        partial_optimizer = partial(Adam, lr=0.001)
        return partial_optimizer

    @pytest.fixture
    def model_wrappers_dict(self, forward_dynamics, policy_net, value_net, device):
        d = ModelWrappersDict(
            {
                ModelNames.FORWARD_DYNAMICS: ModelWrapper(forward_dynamics, device, True),
                ModelNames.POLICY: ModelWrapper(policy_net, device, True),
                ModelNames.VALUE: ModelWrapper(value_net, device, True),
            }
        )
        d.send_to_default_device()
        return d

    @pytest.fixture
    def logger(self, tmp_path):
        return StepIntervalLogger(f"{tmp_path}/tensorboard", 1)

    @pytest.fixture
    def trainer(
        self,
        partial_dataloader,
        partial_policy_optimizer,
        partial_value_optimizer,
        device,
        model_wrappers_dict,
        data_users_dict,
        logger,
    ):
        trainer = DreamingPolicyValueTrainer(
            partial_dataloader,
            partial_policy_optimizer,
            partial_value_optimizer,
            device,
            logger,
            imagination_trajectory_length=8,
            minimum_new_data_count=4,
        )
        trainer.attach_model_wrappers_dict(model_wrappers_dict)
        trainer.attach_data_users_dict(data_users_dict)
        return trainer

    def test_run(self, trainer: DreamingPolicyValueTrainer):
        trainer.run()

    def test_is_trainable(self, trainer: DreamingPolicyValueTrainer) -> None:
        assert trainer.is_trainable() is True
        trainer.initial_states_data_user.clear()
        assert trainer.is_trainable() is False

    def test_is_new_data_available(self, trainer: DreamingPolicyValueTrainer):
        trainer.initial_states_data_user.update()
        assert trainer._is_new_data_available() is True
        trainer.run()
        assert trainer._is_new_data_available() is False

    def test_save_and_load_state(self, trainer: DreamingPolicyValueTrainer, tmp_path, mocker) -> None:
        trainer_path = tmp_path / "dreamer"
        trainer.save_state(trainer_path)
        assert trainer_path.exists()
        assert (trainer_path / "policy_optimizer.pt").exists()
        assert (trainer_path / "value_optimizer.pt").exists()
        assert (trainer_path / "logger.pt").exists()
        logger_state = trainer.logger.state_dict()

        mocked_logger_load_state_dict = mocker.spy(trainer.logger, "load_state_dict")
        trainer.policy_optimizer_state.clear()
        trainer.value_optimizer_state.clear()
        assert trainer.policy_optimizer_state == {}
        assert trainer.value_optimizer_state == {}
        trainer.load_state(trainer_path)
        assert trainer.policy_optimizer_state != {}
        assert trainer.value_optimizer_state != {}
        mocked_logger_load_state_dict.assert_called_once_with(logger_state)
