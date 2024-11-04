import pytest
import torch
from pytest_mock import MockerFixture

from ami.interactions.agents.observation_collecting_agent import (
    DataKeys,
    ObservationCollectingAgent,
)


class TestObservationCollectingAgent:
    @pytest.fixture
    def agent(self, inference_wrappers_dict, data_collectors_dict):
        agent = ObservationCollectingAgent()
        agent.attach_data_collectors(data_collectors_dict)
        agent.attach_inference_models(inference_wrappers_dict)
        return agent

    def test_step(self, agent: ObservationCollectingAgent, mocker: MockerFixture):

        mock_collect = mocker.spy(agent.data_collectors, "collect")
        obs = torch.randn(10)
        assert agent.step(obs) is None
        assert torch.equal(obs, agent.step_data[DataKeys.OBSERVATION])
        mock_collect.assert_called_once()
