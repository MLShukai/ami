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
from ami.interactions.agents.unimodal_encoding_agent import UnimodalEncodingAgent
from ami.utils import Modality

# Test constants
OBSERVATION_DIM = 64
ACTION_DIM = 8


class TestMultimodalTemporalCuriosityAgent:
    @pytest.fixture
    def mock_unimodal_agents(self, mocker: MockerFixture):
        """Create mock unimodal agents for each modality."""
        agents = {}
        # Using actual Modality enum values
        for modality in [Modality.IMAGE, Modality.AUDIO]:
            mock_agent = mocker.create_autospec(UnimodalEncodingAgent, instance=True)
            mock_agent.step.return_value = torch.randn(OBSERVATION_DIM)
            agents[modality] = mock_agent
        return agents

    @pytest.fixture
    def mock_temporal_agent(self, mocker: MockerFixture):
        """Create mock temporal encoding agent."""
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
    def agent_without_action(self, mock_unimodal_agents, mock_temporal_agent, mock_curiosity_agent):
        """Create agent instance without action modality."""
        return MultimodalTemporalCuriosityAgent(
            unimodal_agents=mock_unimodal_agents,
            temporal_agent=mock_temporal_agent,
            curiosity_agent=mock_curiosity_agent,
            include_action_modality=False,
            initial_action=None,
        )

    @pytest.fixture
    def agent_with_action(self, mock_unimodal_agents, mock_temporal_agent, mock_curiosity_agent):
        """Create agent instance with action modality."""
        initial_action = torch.randn(ACTION_DIM)
        return MultimodalTemporalCuriosityAgent(
            unimodal_agents=mock_unimodal_agents,
            temporal_agent=mock_temporal_agent,
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

    def test_initialization_invalid_config(self, mock_unimodal_agents, mock_temporal_agent, mock_curiosity_agent):
        """Test initialization with invalid configuration."""
        with pytest.raises(ValueError):
            MultimodalTemporalCuriosityAgent(
                unimodal_agents=mock_unimodal_agents,
                temporal_agent=mock_temporal_agent,
                curiosity_agent=mock_curiosity_agent,
                include_action_modality=True,
                initial_action=None,
            )

    def test_step_without_action(self, agent_without_action, sample_observation):
        """Test step function without action modality."""
        action = agent_without_action.step(sample_observation)

        # Verify action shape
        assert isinstance(action, torch.Tensor)
        assert action.shape == (ACTION_DIM,)

        # Verify each unimodal agent was called
        for modality, agent in agent_without_action.unimodal_agents.items():
            agent.step.assert_called_once_with(sample_observation[modality])

        # Verify temporal and curiosity agents were called
        agent_without_action.temporal_agent.step.assert_called_once()
        agent_without_action.curiosity_agent.step.assert_called_once()

    def test_step_with_action(self, agent_with_action, sample_observation):
        """Test step function with action modality."""
        initial_action = agent_with_action._previous_action.clone()
        action = agent_with_action.step(sample_observation)

        # Verify action shape and previous action update
        assert isinstance(action, torch.Tensor)
        assert action.shape == (ACTION_DIM,)
        assert not torch.equal(agent_with_action._previous_action, initial_action)
        assert torch.equal(agent_with_action._previous_action, action)

        # Verify temporal agent received correct input including action
        temporal_input = agent_with_action.temporal_agent.step.call_args[0][0]
        assert Modality.ACTION in temporal_input
        assert torch.equal(temporal_input[Modality.ACTION], initial_action)

    def test_save_and_load_state(self, agent_with_action, tmp_path):
        """Test state saving and loading functionality."""
        save_path = tmp_path / "agent_state"

        # Save state
        agent_with_action.save_state(save_path)

        # Verify component agents' save_state was called
        for agent in agent_with_action.unimodal_agents.values():
            agent.save_state.assert_called_once()
        agent_with_action.temporal_agent.save_state.assert_called_once()
        agent_with_action.curiosity_agent.save_state.assert_called_once()

        # Verify previous_action was saved
        assert (save_path / "previous_action.pt").exists()

        # Load state
        new_agent = MultimodalTemporalCuriosityAgent(
            unimodal_agents=agent_with_action.unimodal_agents,
            temporal_agent=agent_with_action.temporal_agent,
            curiosity_agent=agent_with_action.curiosity_agent,
            include_action_modality=True,
            initial_action=torch.randn(ACTION_DIM),  # Different initial action
        )
        new_agent.load_state(save_path)

        # Verify component agents' load_state was called
        for agent in new_agent.unimodal_agents.values():
            agent.load_state.assert_called_once()
        new_agent.temporal_agent.load_state.assert_called_once()
        new_agent.curiosity_agent.load_state.assert_called_once()

        # Verify previous_action was loaded correctly
        assert torch.equal(new_agent._previous_action, agent_with_action._previous_action)

    def test_setup_and_teardown(self, agent_with_action):
        """Test setup and teardown methods."""
        # Test setup
        agent_with_action.setup()
        for agent in agent_with_action.unimodal_agents.values():
            agent.setup.assert_called_once()
        agent_with_action.temporal_agent.setup.assert_called_once()
        agent_with_action.curiosity_agent.setup.assert_called_once()

        # Test teardown
        agent_with_action.teardown()
        for agent in agent_with_action.unimodal_agents.values():
            agent.teardown.assert_called_once()
        agent_with_action.temporal_agent.teardown.assert_called_once()
        agent_with_action.curiosity_agent.teardown.assert_called_once()
