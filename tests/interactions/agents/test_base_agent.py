from tests.helpers import AgentImpl


class TestBaseAgent:
    def test_attach_inference_models(self, inference_wrappers_dict):
        agent = AgentImpl()
        agent.attach_inference_models(inference_wrappers_dict)

        assert agent.model1

    def test_attach_data_collectors(self, data_collectors_dict):
        agent = AgentImpl()
        agent.attach_data_collectors(data_collectors_dict)

        assert agent.data_collector1
