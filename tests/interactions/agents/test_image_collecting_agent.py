import pytest
import torch
from pytest_mock import MockerFixture

from ami.interactions.agents.image_collecting_agent import (
    DataKeys,
    ImageCollectingAgent,
)


class TestImageCollectingAgent:
    @pytest.fixture
    def agent(self, inference_wrappers_dict, data_collectors_dict):
        agent = ImageCollectingAgent()
        agent.attach_data_collectors(data_collectors_dict)
        agent.attach_inference_models(inference_wrappers_dict)
        return agent

    def test_step(self, agent: ImageCollectingAgent, mocker: MockerFixture):

        mock_collect = mocker.spy(agent.data_collectors, "collect")
        obs = torch.randn(10)
        assert agent.step(obs) is None
        assert torch.equal(obs, agent.step_data[DataKeys.OBSERVATION])
        mock_collect.assert_called_once()
