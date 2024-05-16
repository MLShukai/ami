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
        hidden_shape,
        action_shape,
        num_collect,
        use_embed_obs_as_observation,
        embed_obs_shape,
        """,
        [
            (128, 0.99, 0.99, (64, 64), (4, 64), (64, 64), 256, False, (128,)),
            (64, 0.98, 0.999, (3, 84, 84), (8, 128), (5,), 32, True, (32,)),
        ],
    )
    def test_make_dataset(
        self,
        max_size,
        gamma,
        gae_lambda,
        observation_shape,
        hidden_shape,
        action_shape,
        num_collect,
        use_embed_obs_as_observation,
        embed_obs_shape,
    ):
        buffer = PPOTrajectoryBuffer.reconstructable_init(max_size, gamma, gae_lambda, use_embed_obs_as_observation)

        assert buffer.dataset_size == 0

        for _ in range(num_collect):
            step_data = StepData()
            step_data[DataKeys.OBSERVATION] = torch.randn(*observation_shape)
            step_data[DataKeys.EMBED_OBSERVATION] = torch.randn(*embed_obs_shape)
            step_data[DataKeys.HIDDEN] = torch.randn(*hidden_shape)
            step_data[DataKeys.ACTION] = torch.randn(*action_shape)
            step_data[DataKeys.ACTION_LOG_PROBABILITY] = torch.randn(*action_shape)
            step_data[DataKeys.REWARD] = torch.randn(1)
            step_data[DataKeys.VALUE] = torch.randn(1)

            buffer.add(step_data)

        dataset = buffer.make_dataset()

        length = min(max_size, num_collect) - 1
        assert buffer.dataset_size == length

        observations, hiddens, actions, logprobs, advantages, returns, values = dataset[0:length]
        if use_embed_obs_as_observation:
            assert observations.size() == (length, *embed_obs_shape)
        else:
            assert observations.size() == (length, *observation_shape)
        assert hiddens.size() == (length, *hidden_shape)
        assert actions.size() == (length, *action_shape)
        assert logprobs.size() == (length, *action_shape)
        assert advantages.size() == (length, 1)
        assert returns.size() == (length, 1)
        assert values.size() == (length, 1)
