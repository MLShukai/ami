import pytest
import torch
from pytest_mock import MockerFixture

from ami.interactions.agents.curiosity_agent import CuriosityAgent
from ami.interactions.agents.multimodal_temporal_curiosity_agent import (
    MultimodalTemporalCuriosityAgent,
)
from ami.interactions.agents.multimodal_temporal_encoding_agent import (
    MultimodalTemporalEncodingAgent,
)
from ami.utils import Modality

# Test constants
OBSERVATION_DIM = 64
ACTION_DIM = 8


class TestMultimodalTemporalCuriosityAgent:
    @pytest.fixture
    def mock_multimodal_temporal_agent(self, mocker: MockerFixture):
        """Create mock multimodal temporal encoding agent."""
        mock_agent = mocker.create_autospec(MultimodalTemporalEncodingAgent, instance=True)
        mock_agent.step.return_value = torch.randn(OBSERVATION_DIM)
        return mock_agent

    @pytest.fixture
    def mock_curiosity_agent(self, mocker: MockerFixture):
        """Create mock curiosity agent."""
        mock_agent = mocker.create_autospec(CuriosityAgent, instance=True)
        mock_agent.step.return_value = torch.randn(ACTION_DIM)
        return mock_agent

    @pytest.fixture
    def agent_without_action(self, mock_multimodal_temporal_agent, mock_curiosity_agent):
        """Create agent instance without action modality."""
        return MultimodalTemporalCuriosityAgent(
            multimodal_temporal_agent=mock_multimodal_temporal_agent,
            curiosity_agent=mock_curiosity_agent,
            include_action_modality=False,
            initial_action=None,
        )

    @pytest.fixture
    def agent_with_action(self, mock_multimodal_temporal_agent, mock_curiosity_agent):
        """Create agent instance with action modality."""
        initial_action = torch.randn(ACTION_DIM)
        return MultimodalTemporalCuriosityAgent(
            multimodal_temporal_agent=mock_multimodal_temporal_agent,
            curiosity_agent=mock_curiosity_agent,
            include_action_modality=True,
            initial_action=initial_action,
        )

    @pytest.fixture
    def sample_observation(self):
        """Create sample observation for testing."""
        return {Modality.IMAGE: torch.randn(OBSERVATION_DIM), Modality.AUDIO: torch.randn(OBSERVATION_DIM)}

    def test_initialization_without_action(self, agent_without_action):
        """Test initialization without action modality."""
        assert not agent_without_action._include_action_modality
        assert agent_without_action._previous_action is None

    def test_initialization_with_action(self, agent_with_action):
        """Test initialization with action modality."""
        assert agent_with_action._include_action_modality
        assert agent_with_action._previous_action is not None
        assert agent_with_action._previous_action.shape == (ACTION_DIM,)

    def test_initialization_invalid_config(self, mock_multimodal_temporal_agent, mock_curiosity_agent):
        """Test initialization with invalid configuration."""
        with pytest.raises(ValueError):
            MultimodalTemporalCuriosityAgent(
                multimodal_temporal_agent=mock_multimodal_temporal_agent,
                curiosity_agent=mock_curiosity_agent,
                include_action_modality=True,
                initial_action=None,
            )

    def test_step_without_action(self, agent_without_action, sample_observation):
        """Test step function without action modality."""
        # Store original observation for comparison
        orig_observation = {k: v.clone() for k, v in sample_observation.items()}

        action = agent_without_action.step(sample_observation)

        # Verify action shape
        assert isinstance(action, torch.Tensor)
        assert action.shape == (ACTION_DIM,)

        # Verify multimodal_temporal_agent was called with correct input
        agent_without_action.multimodal_temporal_agent.step.assert_called_once()
        multimodal_input = agent_without_action.multimodal_temporal_agent.step.call_args[0][0]
        assert Modality.ACTION not in multimodal_input
        for modality in orig_observation:
            assert torch.equal(multimodal_input[modality], orig_observation[modality])

        # Verify curiosity agent was called with encoded observation
        agent_without_action.curiosity_agent.step.assert_called_once()
        encoded_input = agent_without_action.curiosity_agent.step.call_args[0][0]
        assert torch.equal(encoded_input, agent_without_action.multimodal_temporal_agent.step.return_value)

    def test_step_with_action(self, agent_with_action, sample_observation):
        """Test step function with action modality."""
        initial_action = agent_with_action._previous_action.clone()
        orig_observation = {k: v.clone() for k, v in sample_observation.items()}

        action = agent_with_action.step(sample_observation)

        # Verify action shape and previous action update
        assert isinstance(action, torch.Tensor)
        assert action.shape == (ACTION_DIM,)
        assert not torch.equal(agent_with_action._previous_action, initial_action)
        assert torch.equal(agent_with_action._previous_action, action)

        # Verify multimodal_temporal_agent received correct input including action
        agent_with_action.multimodal_temporal_agent.step.assert_called_once()
        multimodal_input = agent_with_action.multimodal_temporal_agent.step.call_args[0][0]
        assert Modality.ACTION in multimodal_input
        assert torch.equal(multimodal_input[Modality.ACTION], initial_action)
        for modality in orig_observation:
            assert torch.equal(multimodal_input[modality], orig_observation[modality])

        # Verify curiosity agent was called with encoded observation
        agent_with_action.curiosity_agent.step.assert_called_once()
        encoded_input = agent_with_action.curiosity_agent.step.call_args[0][0]
        assert torch.equal(encoded_input, agent_with_action.multimodal_temporal_agent.step.return_value)

    def test_save_and_load_state(self, agent_with_action, tmp_path):
        """Test state saving and loading functionality."""
        save_path = tmp_path / "agent_state"

        # Save state
        agent_with_action.save_state(save_path)

        # Verify component agents' save_state was called with correct paths
        agent_with_action.multimodal_temporal_agent.save_state.assert_called_once_with(
            save_path / "multimodal_temporal"
        )
        agent_with_action.curiosity_agent.save_state.assert_called_once_with(save_path / "curiosity")

        # Verify previous_action was saved
        assert (save_path / "previous_action.pt").exists()

        # Load state
        new_agent = MultimodalTemporalCuriosityAgent(
            multimodal_temporal_agent=agent_with_action.multimodal_temporal_agent,
            curiosity_agent=agent_with_action.curiosity_agent,
            include_action_modality=True,
            initial_action=torch.randn(ACTION_DIM),  # Different initial action
        )
        new_agent.load_state(save_path)

        # Verify component agents' load_state was called with correct paths
        new_agent.multimodal_temporal_agent.load_state.assert_called_once_with(save_path / "multimodal_temporal")
        new_agent.curiosity_agent.load_state.assert_called_once_with(save_path / "curiosity")

        # Verify previous_action was loaded correctly
        assert torch.equal(new_agent._previous_action, agent_with_action._previous_action)

    def test_setup_and_teardown(self, agent_with_action):
        """Test setup and teardown methods."""
        # Test setup
        agent_with_action.setup()
        agent_with_action.multimodal_temporal_agent.setup.assert_called_once()
        agent_with_action.curiosity_agent.setup.assert_called_once()

        # Test teardown
        agent_with_action.teardown()
        agent_with_action.multimodal_temporal_agent.teardown.assert_called_once()
        agent_with_action.curiosity_agent.teardown.assert_called_once()
