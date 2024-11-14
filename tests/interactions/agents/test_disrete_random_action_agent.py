import pytest
import torch

from ami.interactions.agents.discrete_random_action_agent import (
    DiscreteRandomActionAgent,
)


class TestDiscreteRandomActionAgent:
    @pytest.fixture
    def agent(self):
        action_choices = [3, 2, 4]  # 3 categories with 3, 2, and 4 choices respectively
        return DiscreteRandomActionAgent(action_choices, min_action_repeat=1, max_action_repeat=3)

    def test_initialization(self, agent):
        assert agent.num_actions == 3
        assert agent.action_choices_per_category == [3, 2, 4]
        assert agent.min_action_repeat == 1
        assert agent.max_action_repeat == 3

    def test_setup(self, agent):
        agent.setup(None)
        assert len(agent.remaining_action_repeat_counts) == 3
        assert len(agent.action) == 3
        assert all(count == 0 for count in agent.remaining_action_repeat_counts)
        assert all(action == 0 for action in agent.action)

    def test_step_output_shape(self, agent):
        agent.setup(None)
        action = agent.step(None)
        assert isinstance(action, torch.Tensor)
        assert action.shape == (3,)
        assert action.dtype == torch.long

    def test_action_choices(self, agent):
        agent.setup(None)
        for _ in range(100):  # Run multiple times to increase chance of covering all possibilities
            action = agent.step(None)
            assert 0 <= action[0] < 3
            assert 0 <= action[1] < 2
            assert 0 <= action[2] < 4
