import pytest
import torch

from ami.data.utils import DataCollectorsDict
from ami.interactions.agents.image_encoding_agent import DataKeys, ImageEncodingAgent
from ami.models.components.small_conv_net import SmallConvNet
from ami.models.model_wrapper import ModelWrapper
from ami.models.utils import InferenceWrappersDict, ModelNames, ModelWrappersDict


class TestImageEncodingAgent:
    @pytest.fixture
    def agent(self) -> ImageEncodingAgent:

        agent = ImageEncodingAgent()
        md = ModelWrappersDict(
            {
                ModelNames.IMAGE_ENCODER: ModelWrapper(SmallConvNet(64, 64, 3, 256), "cpu"),
            }
        )
        dd = DataCollectorsDict()
        agent.attach_inference_models(md.inference_wrappers_dict)
        agent.attach_data_collectors(dd)

        return agent

    def test_step(self, agent: ImageEncodingAgent) -> None:
        obs = torch.randn(3, 64, 64)

        out = agent.step(obs)
        assert out.shape == (256,)
        assert DataKeys.OBSERVATION in agent.step_data
        assert DataKeys.EMBED_OBSERVATION in agent.step_data
