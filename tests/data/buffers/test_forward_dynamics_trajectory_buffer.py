import pytest
import torch

from ami.data.buffers.forward_dynamics_trajectory_buffer import (
    ForwardDynamicsTrajectoryBuffer,
)
from ami.data.step_data import DataKeys, StepData


class TestForwardDynamicsBuffer:
    @pytest.mark.parametrize(
        """
        max_size,
        embed_observation_shape,
        action_shape,
        hidden_shape,
        num_collect,
        """,
        [(128, (64, 64), (64, 64), (8, 32), 256), (64, (3, 84, 84), (5,), (4, 16), 128)],
    )
    def test_make_dataset(self, max_size, embed_observation_shape, action_shape, hidden_shape, num_collect):
        buffer = ForwardDynamicsTrajectoryBuffer.reconstructable_init(max_size)

        for _ in range(num_collect):
            step_data = StepData()
            step_data[DataKeys.EMBED_OBSERVATION] = torch.randn(*embed_observation_shape)
            step_data[DataKeys.NEXT_EMBED_OBSERVATION] = torch.randn(*embed_observation_shape)
            step_data[DataKeys.ACTION] = torch.randn(*action_shape)
            step_data[DataKeys.HIDDEN] = torch.randn(*hidden_shape)

            buffer.add(step_data)

        dataset = buffer.make_dataset()

        length = min(max_size, num_collect)
        assert len(buffer) == length

        embed_observations, next_embed_observations, actions, hiddens = dataset[0:length]
        assert embed_observations.size() == (length, *embed_observation_shape)
        assert next_embed_observations.size() == (length, *embed_observation_shape)
        assert actions.size() == (length, *action_shape)
        assert hiddens.size() == (length, *hidden_shape)
