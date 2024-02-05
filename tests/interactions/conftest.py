from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from ami.interactions.interaction import BaseAgent, BaseEnvironment


@pytest.fixture
def mock_env(mocker: MockerFixture) -> Mock:
    env = mocker.Mock(spec=BaseEnvironment)
    env.observe.return_value = "observation"
    return env


@pytest.fixture
def mock_agent(mocker: MockerFixture) -> Mock:
    agent = mocker.Mock(spec=BaseAgent)
    agent.step.return_value = "action"
    agent.setup.return_value = "setup_action"
    agent.teardown.return_value = "teardown_action"
    return agent
