from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from ami.interactions.agents.base_agent import BaseAgent
from ami.interactions.environments.base_environment import BaseEnvironment
from ami.interactions.simple_interation import SimpleInteraction


class TestSimpleInteraction:
    @pytest.fixture
    def mock_env(self, mocker: MockerFixture) -> Mock:
        env = mocker.Mock(spec=BaseEnvironment)
        env.observe.return_value = "observation"
        return env

    @pytest.fixture
    def mock_agent(self, mocker: MockerFixture) -> Mock:
        agent = mocker.Mock(spec=BaseAgent)
        agent.step.return_value = "action"
        agent.setup.return_value = "setup_action"
        agent.teardown.return_value = "teardown_action"
        return agent

    @pytest.fixture
    def simple_interaction(self, mock_env, mock_agent) -> SimpleInteraction[str, str]:
        return SimpleInteraction(mock_env, mock_agent)

    def test_setup(self, mock_env: Mock, mock_agent: Mock) -> None:
        simple_interaction = SimpleInteraction(mock_env, mock_agent)
        simple_interaction.setup()

        mock_env.setup.assert_called_once()
        mock_agent.setup.assert_called_once_with("observation")
        mock_env.affect.assert_called_once_with("setup_action")

    def test_step(self, mock_env: Mock, mock_agent: Mock) -> None:
        simple_interaction = SimpleInteraction(mock_env, mock_agent)
        simple_interaction.step()

        mock_agent.step.assert_called_once_with("observation")
        mock_env.affect.assert_called_once_with("action")

    def test_teardown(self, mock_env: Mock, mock_agent: Mock) -> None:
        simple_interaction = SimpleInteraction(mock_env, mock_agent)
        simple_interaction.teardown()

        mock_agent.teardown.assert_called_once_with("observation")
        mock_env.affect.assert_called_once_with("teardown_action")
        mock_env.teardown.assert_called_once()
