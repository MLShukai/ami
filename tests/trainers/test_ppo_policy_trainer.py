from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.causal_data_buffer import CausalDataBuffer
from ami.data.buffers.ppo_trajectory_buffer import PPOTrajectoryBuffer
from ami.data.step_data import DataKeys, StepData
from ami.data.utils import DataCollectorsDict, DataUsersDict
from ami.models.components.discrete_policy_head import DiscretePolicyHead
from ami.models.components.fully_connected_value_head import FullyConnectedValueHead
from ami.models.components.sioconvps import SioConvPS
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.policy_value_common_net import (
    PolicyValueCommonNet,
    PrimitivePolicyValueCommonNet,
    TemporalPolicyValueCommonNet,
)
from ami.models.utils import ModelWrappersDict
from ami.tensorboard_loggers import StepIntervalLogger
from ami.trainers.ppo_policy_trainer import (
    PPOPolicyTrainer,
    PPOPrimitivePolicyTrainer,
    PPOTemporalPolicyTrainer,
    RandomTimeSeriesSampler,
)
from tests.models.test_policy_value_common_net import CatObsHidden


class TestPPOPolicyTrainer:
    @pytest.fixture
    def partial_dataloader(self):
        partial_dataloader = partial(DataLoader, batch_size=2, shuffle=True)
        return partial_dataloader

    @pytest.fixture
    def partial_optimizer(self):
        partial_optimizer = partial(Adam, lr=0.001)
        return partial_optimizer

    @pytest.fixture
    def policy_value(self):
        obs_layer = nn.Linear(128, 64)
        hidden_layer = nn.Linear(256, 128)
        obs_hidden_proj = CatObsHidden()
        core_model = nn.Linear(128 + 64, 16)
        policy = DiscretePolicyHead(16, [8])
        value = FullyConnectedValueHead(16)

        return PolicyValueCommonNet(obs_layer, hidden_layer, obs_hidden_proj, core_model, policy, value)

    @pytest.fixture
    def policy_value_wrappers_dict(
        self,
        device: torch.device,
        policy_value,
    ) -> ModelWrappersDict:
        d = ModelWrappersDict({ModelNames.POLICY_VALUE: ModelWrapper(policy_value, device, True)})
        d.send_to_default_device()
        return d

    @pytest.fixture
    def trajectory_step_data(self) -> StepData:
        return StepData(
            {
                DataKeys.OBSERVATION: torch.randn(128),
                DataKeys.HIDDEN: torch.randn(256),
                DataKeys.ACTION: torch.zeros((1,), dtype=torch.int64),
                DataKeys.ACTION_LOG_PROBABILITY: torch.randn(1),
                DataKeys.REWARD: torch.randn(1),
                DataKeys.VALUE: torch.randn(1),
            }
        )

    @pytest.fixture
    def trajectory_buffer_dict(self, trajectory_step_data) -> DataCollectorsDict:
        d = DataCollectorsDict.from_data_buffers(
            **{BufferNames.PPO_TRAJECTORY: PPOTrajectoryBuffer.reconstructable_init(32)}
        )
        for _ in range(8):
            d.collect(trajectory_step_data)
        return d

    @pytest.fixture
    def logger(self, tmp_path):
        return StepIntervalLogger(f"{tmp_path}/tensorboard", 1)

    @pytest.fixture
    def trainer(
        self,
        partial_dataloader,
        partial_optimizer,
        policy_value_wrappers_dict,
        trajectory_buffer_dict: DataCollectorsDict,
        device,
        logger,
    ) -> PPOPolicyTrainer:
        trainer = PPOPolicyTrainer(partial_dataloader, partial_optimizer, device, logger, max_epochs=1)
        trainer.attach_model_wrappers_dict(policy_value_wrappers_dict)
        trainer.attach_data_users_dict(trajectory_buffer_dict.get_data_users())
        return trainer

    def test_run(self, trainer: PPOPolicyTrainer) -> None:
        assert trainer.is_trainable() is True
        trainer.run()
        assert trainer.is_trainable() is False

    def test_save_and_load_state(self, trainer: PPOPolicyTrainer, tmp_path, mocker) -> None:
        trainer_path = tmp_path / "ppo_policy"
        trainer.save_state(trainer_path)
        assert trainer_path.exists()
        assert (trainer_path / "optimizer.pt").exists()
        assert (trainer_path / "logger.pt").exists()
        logger_state = trainer.logger.state_dict()

        mocked_logger_load_state_dict = mocker.spy(trainer.logger, "load_state_dict")
        trainer.optimizer_state.clear()
        assert trainer.optimizer_state == {}
        trainer.load_state(trainer_path)
        assert trainer.optimizer_state != {}
        mocked_logger_load_state_dict.assert_called_once_with(logger_state)


class TestPPOPrimitivePolicyTrainer:
    @pytest.fixture
    def partial_dataloader(self):
        partial_dataloader = partial(DataLoader, batch_size=2)
        return partial_dataloader

    @pytest.fixture
    def partial_optimizer(self):
        partial_optimizer = partial(Adam, lr=0.001)
        return partial_optimizer

    @pytest.fixture
    def policy_value(self):
        return PrimitivePolicyValueCommonNet(
            observation_projection=nn.Linear(128, 64),
            core_model=nn.Linear(64, 16),
            policy_head=DiscretePolicyHead(16, [8]),
            value_head=FullyConnectedValueHead(16),
        )

    @pytest.fixture
    def policy_value_wrappers_dict(
        self,
        device: torch.device,
        policy_value,
    ) -> ModelWrappersDict:
        d = ModelWrappersDict({ModelNames.POLICY_VALUE: ModelWrapper(policy_value, device, True)})
        d.send_to_default_device()
        return d

    @pytest.fixture
    def trajectory_step_data(self) -> StepData:
        return StepData(
            {
                DataKeys.OBSERVATION: torch.randn(128),
                DataKeys.ACTION: torch.zeros((1,), dtype=torch.int64),
                DataKeys.ACTION_LOG_PROBABILITY: torch.randn(1),
                DataKeys.REWARD: torch.randn(1),
                DataKeys.VALUE: torch.randn(1),
            }
        )

    @pytest.fixture
    def trajectory_buffer_dict(self, trajectory_step_data) -> DataCollectorsDict:
        d = DataCollectorsDict.from_data_buffers(
            **{
                BufferNames.PPO_TRAJECTORY: CausalDataBuffer.reconstructable_init(
                    32,
                    [
                        DataKeys.OBSERVATION,
                        DataKeys.ACTION,
                        DataKeys.ACTION_LOG_PROBABILITY,
                        DataKeys.REWARD,
                        DataKeys.VALUE,
                    ],
                )
            }
        )
        for _ in range(8):
            d.collect(trajectory_step_data)
        return d

    @pytest.fixture
    def logger(self, tmp_path):
        return StepIntervalLogger(f"{tmp_path}/tensorboard", 1)

    @pytest.fixture
    def trainer(
        self,
        partial_dataloader,
        partial_optimizer,
        policy_value_wrappers_dict,
        trajectory_buffer_dict: DataCollectorsDict,
        device,
        logger,
    ) -> PPOPrimitivePolicyTrainer:
        trainer = PPOPrimitivePolicyTrainer(
            partial_dataloader,
            partial(RandomTimeSeriesSampler, sequence_length=2),
            partial_optimizer,
            device,
            logger,
        )
        trainer.attach_model_wrappers_dict(policy_value_wrappers_dict)
        trainer.attach_data_users_dict(trajectory_buffer_dict.get_data_users())
        return trainer

    def test_run(self, trainer: PPOPolicyTrainer) -> None:
        assert trainer.is_trainable() is True
        trainer.run()
        assert trainer.is_trainable() is False

    def test_save_and_load_state(self, trainer: PPOPolicyTrainer, tmp_path, mocker) -> None:
        trainer_path = tmp_path / "ppo_primitive_policy"
        trainer.save_state(trainer_path)
        assert trainer_path.exists()
        assert (trainer_path / "optimizer.pt").exists()
        assert (trainer_path / "logger.pt").exists()
        logger_state = trainer.logger.state_dict()

        mocked_logger_load_state_dict = mocker.spy(trainer.logger, "load_state_dict")
        trainer.optimizer_state.clear()
        assert trainer.optimizer_state == {}
        trainer.load_state(trainer_path)
        assert trainer.optimizer_state != {}
        mocked_logger_load_state_dict.assert_called_once_with(logger_state)


class TestPPOTemporalPolicyTrainer:
    @pytest.fixture
    def partial_dataloader(self):
        partial_dataloader = partial(DataLoader, batch_size=2)
        return partial_dataloader

    @pytest.fixture
    def partial_optimizer(self):
        partial_optimizer = partial(Adam, lr=0.001)
        return partial_optimizer

    @pytest.fixture
    def policy_value(self):
        return TemporalPolicyValueCommonNet(
            observation_flatten=nn.Linear(128, 64),
            core_model=SioConvPS(3, 64, 128, 0.1),
            policy_head=DiscretePolicyHead(64, [8]),
            value_head=FullyConnectedValueHead(64),
        )

    @pytest.fixture
    def policy_value_wrappers_dict(
        self,
        device: torch.device,
        policy_value,
    ) -> ModelWrappersDict:
        d = ModelWrappersDict({ModelNames.POLICY_VALUE: ModelWrapper(policy_value, device, True)})
        d.send_to_default_device()
        return d

    @pytest.fixture
    def trajectory_step_data(self) -> StepData:
        return StepData(
            {
                DataKeys.OBSERVATION: torch.randn(128),
                DataKeys.HIDDEN: torch.randn(3, 64),
                DataKeys.ACTION: torch.zeros((1,), dtype=torch.int64),
                DataKeys.ACTION_LOG_PROBABILITY: torch.randn(1),
                DataKeys.REWARD: torch.randn(1).squeeze(),
                DataKeys.VALUE: torch.randn(1).squeeze(),
            }
        )

    @pytest.fixture
    def trajectory_buffer_dict(self, trajectory_step_data) -> DataCollectorsDict:
        d = DataCollectorsDict.from_data_buffers(
            **{
                BufferNames.PPO_TRAJECTORY: CausalDataBuffer.reconstructable_init(
                    32,
                    [
                        DataKeys.OBSERVATION,
                        DataKeys.HIDDEN,
                        DataKeys.ACTION,
                        DataKeys.ACTION_LOG_PROBABILITY,
                        DataKeys.REWARD,
                        DataKeys.VALUE,
                    ],
                )
            }
        )
        for _ in range(8):
            d.collect(trajectory_step_data)
        return d

    @pytest.fixture
    def logger(self, tmp_path):
        return StepIntervalLogger(f"{tmp_path}/tensorboard", 1)

    @pytest.fixture
    def trainer(
        self,
        partial_dataloader,
        partial_optimizer,
        policy_value_wrappers_dict,
        trajectory_buffer_dict: DataCollectorsDict,
        device,
        logger,
    ) -> PPOTemporalPolicyTrainer:
        trainer = PPOTemporalPolicyTrainer(
            partial_dataloader,
            partial(RandomTimeSeriesSampler, sequence_length=3),
            partial_optimizer,
            device,
            logger,
        )
        trainer.attach_model_wrappers_dict(policy_value_wrappers_dict)
        trainer.attach_data_users_dict(trajectory_buffer_dict.get_data_users())
        return trainer

    def test_run(self, trainer: PPOTemporalPolicyTrainer) -> None:
        assert trainer.is_trainable() is True
        trainer.run()
        assert trainer.is_trainable() is False

    def test_save_and_load_state(self, trainer: PPOTemporalPolicyTrainer, tmp_path, mocker) -> None:
        trainer_path = tmp_path / "ppo_temporal_policy"
        trainer.save_state(trainer_path)
        assert trainer_path.exists()
        assert (trainer_path / "optimizer.pt").exists()
        assert (trainer_path / "logger.pt").exists()
        logger_state = trainer.logger.state_dict()

        mocked_logger_load_state_dict = mocker.spy(trainer.logger, "load_state_dict")
        trainer.optimizer_state.clear()
        assert trainer.optimizer_state == {}
        trainer.load_state(trainer_path)
        assert trainer.optimizer_state != {}
        mocked_logger_load_state_dict.assert_called_once_with(logger_state)
