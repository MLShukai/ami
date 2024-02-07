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
