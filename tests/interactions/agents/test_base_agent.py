from pytest_mock import MockerFixture

from ami.interactions.agents.base_agent import BaseAgent
from tests.helpers import AgentImpl


class TestBaseAgent:
    def test_attach_inference_models(self, inference_wrappers_dict):
        agent = AgentImpl()
        agent.attach_inference_models(inference_wrappers_dict)

        assert agent.model1
        assert agent.check_model_exists("model1")
        assert agent.check_model_exists("model2")
        assert agent.check_model_exists("model_device")
        assert not agent.check_model_exists("model_no_inference")

    def test_attach_data_collectors(self, data_collectors_dict):
        agent = AgentImpl()
        agent.attach_data_collectors(data_collectors_dict)

        assert agent.data_collector1

    def test_child_agent_method_call(self, mocker: MockerFixture, inference_wrappers_dict, data_collectors_dict):
        mock_agent = mocker.Mock(BaseAgent)

        agent = AgentImpl(mock_agent)
        agent.attach_inference_models(inference_wrappers_dict)
        mock_agent.attach_inference_models.assert_called_once_with(inference_wrappers_dict)

        agent.attach_data_collectors(data_collectors_dict)
        mock_agent.attach_data_collectors.assert_called_once_with(data_collectors_dict)

        agent.setup()
        mock_agent.setup.assert_called_once()

        agent.teardown()
        mock_agent.setup.assert_called_once()
