from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from ami.interactions.interaction import BaseAgent, BaseEnvironment, Interaction


class TestInteraction:
    @pytest.fixture
    def interaction(self, mock_env, mock_agent) -> Interaction:
        return Interaction(mock_env, mock_agent)

    def test_setup(self, mock_env: Mock, mock_agent: Mock) -> None:
        interaction = Interaction(mock_env, mock_agent)
        interaction.setup()

        mock_env.setup.assert_called_once()
        mock_agent.setup.assert_called_once_with("observation")
        mock_env.affect.assert_called_once_with("setup_action")

    def test_step(self, mock_env: Mock, mock_agent: Mock) -> None:
        interaction = Interaction(mock_env, mock_agent)
        interaction.step()

        mock_agent.step.assert_called_once_with("observation")
        mock_env.affect.assert_called_once_with("action")

    def test_teardown(self, mock_env: Mock, mock_agent: Mock) -> None:
        interaction = Interaction(mock_env, mock_agent)
        interaction.teardown()

        mock_agent.teardown.assert_called_once_with("observation")
        mock_env.affect.assert_called_once_with("teardown_action")
        mock_env.teardown.assert_called_once()

    def test_save_and_load_state(self, mock_env: Mock, mock_agent: Mock, tmp_path) -> None:
        interaction = Interaction(mock_env, mock_agent)
        interaction_path = tmp_path / "interaction"
        agent_path = interaction_path / "agent"
        environment_path = interaction_path / "environment"

        interaction.save_state(interaction_path)
        assert interaction_path.exists()
        assert not agent_path.exists()
        assert not environment_path.exists()
        mock_env.save_state.assert_called_once_with(environment_path)
        mock_agent.save_state.assert_called_once_with(agent_path)

        interaction.load_state(interaction_path)
        mock_env.load_state.assert_called_once_with(environment_path)
        mock_agent.load_state.assert_called_once_with(agent_path)

    def test_pause_resume_event_callbacks(self, mock_env: Mock, mock_agent: Mock) -> None:
        interaction = Interaction(mock_env, mock_agent)

        interaction.on_paused()
        mock_env.on_paused.assert_called_once()
        mock_agent.on_paused.assert_called_once()

        interaction.on_resumed()
        mock_env.on_resumed.assert_called_once()
        mock_agent.on_resumed.assert_called_once()
