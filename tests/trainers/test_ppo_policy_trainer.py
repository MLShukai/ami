from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.ppo_trajectory_buffer import PPOTrajectoryBuffer
from ami.data.step_data import DataKeys, StepData
from ami.data.utils import DataCollectorsDict, DataUsersDict
from ami.models.components.discrete_policy_head import DiscretePolicyHead
from ami.models.components.fully_connected_value_head import FullyConnectedValueHead
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.policy_value_common_net import ModularPolicyValueCommonNet
from ami.models.utils import ModelWrappersDict
from ami.trainers.ppo_policy_trainer import PPOPolicyTrainer


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
        base_model = nn.Linear(128, 16)
        policy = DiscretePolicyHead(16, [8])
        value = FullyConnectedValueHead(16)

        return ModularPolicyValueCommonNet(base_model, policy, value)

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
            **{BufferNames.PPO_TRAJECTORY: PPOTrajectoryBuffer.reconstructable_init(32)}
        )
        for _ in range(8):
            d.collect(trajectory_step_data)
        return d

    @pytest.fixture
    def trainer(
        self,
        partial_dataloader,
        partial_optimizer,
        policy_value_wrappers_dict,
        trajectory_buffer_dict: DataCollectorsDict,
        device,
    ) -> PPOPolicyTrainer:
        trainer = PPOPolicyTrainer(partial_dataloader, partial_optimizer, device, max_epochs=1)
        trainer.attach_model_wrappers_dict(policy_value_wrappers_dict)
        trainer.attach_data_users_dict(trajectory_buffer_dict.get_data_users())
        return trainer

    def test_run(self, trainer: PPOPolicyTrainer) -> None:
        assert trainer.is_trainable() is True
        trainer.run()
        assert trainer.is_trainable() is False
