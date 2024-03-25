import pytest
import torch

from ami.data.buffers.ppo_trajectory_buffer import PPOTrajectoryBuffer
from ami.data.step_data import DataKeys, StepData


class TestPPOTrajectoryBuffer:
    @pytest.mark.parametrize(
        """
        max_size,
        gamma,
        gae_lambda,
        observation_shape,
        action_shape,
        num_collect,
        """,
        [(128, 0.99, 0.99, (64, 64), (64, 64), 256), (64, 0.98, 0.999, (3, 84, 84), (5,), 32)],
    )
    def test_make_dataset(self, max_size, gamma, gae_lambda, observation_shape, action_shape, num_collect):
        buffer = PPOTrajectoryBuffer(max_size, gamma, gae_lambda)

        for _ in range(num_collect):
            step_data = StepData()
            step_data[DataKeys.OBSERVATION] = torch.randn(*observation_shape)
            step_data[DataKeys.ACTION] = torch.randn(*action_shape)
            step_data[DataKeys.ACTION_LOG_PROBABILITY] = torch.randn(*action_shape)
            step_data[DataKeys.REWARD] = torch.randn(1)
            step_data[DataKeys.VALUE] = torch.randn(1)

            buffer.add(step_data)

        dataset = buffer.make_dataset()

        length = min(max_size, num_collect) - 1
        observations, actions, logprobs, advantages, returns, values = dataset[0:length]
        assert observations.size() == (length, *observation_shape)
        assert actions.size() == (length, *action_shape)
        assert logprobs.size() == (length, *action_shape)
        assert advantages.size() == (length, 1)
        assert returns.size() == (length, 1)
        assert values.size() == (length, 1)
