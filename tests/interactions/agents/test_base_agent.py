from ami.interactions.agents.base_agent import BaseAgent


class AgentImpl(BaseAgent[str, str]):
    def on_inference_models_attached(self) -> None:
        self.model1 = self.get_inference_model("model1")

    def on_data_collectors_attached(self) -> None:
        self.data_collector1 = self.get_data_collector("buffer1")

    def step(self, observation: str) -> str:
        return "action"


class TestBaseAgent:
    def test_attach_inference_models(self, inference_wrappers_dict):
        agent = AgentImpl()
        agent.attach_inference_models(inference_wrappers_dict)

        assert agent.model1

    def test_attach_data_collectors(self, data_collectors_dict):
        agent = AgentImpl()
        agent.attach_data_collectors(data_collectors_dict)

        assert agent.data_collector1
