import pytest
import torch
import torch.nn as nn

from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.utils import DataCollectorsDict
from ami.interactions.agents.curiosity_image_ppo_agent import (
    BufferNames,
    CuriosityImagePPOAgent,
    DataKeys,
    ForwardDynamics,
    ModelNames,
    PolicyValueCommonNet,
)
from ami.models.components.discrete_policy_head import DiscretePolicyHead
from ami.models.components.fully_connected_fixed_std_normal import (
    FullyConnectedFixedStdNormal,
)
from ami.models.components.fully_connected_value_head import FullyConnectedValueHead
from ami.models.components.multi_embeddings import MultiEmbeddings
from ami.models.components.sconv import SConv
from ami.models.components.small_conv_net import SmallConvNet
from ami.models.utils import InferenceWrappersDict, ModelWrapper, ModelWrappersDict

CHANNELS, WIDTH, HEIGHT = (3, 128, 128)
EMBED_OBS_DIM = 64
ACTION_CHOICES_PER_CATEGORY = [5, 4, 3, 2]
ACTION_DIM = 8
FLATTEN_ACTION_DIM = ACTION_DIM * len(ACTION_CHOICES_PER_CATEGORY)

SCONV_DIM = 64
DEPTH = 2


class TestCuriosityImagePPOAgent:
    @pytest.fixture
    def inference_models(self, device) -> InferenceWrappersDict:
        image_encoder = SmallConvNet(WIDTH, HEIGHT, CHANNELS, EMBED_OBS_DIM)
        forward_dynamics = ForwardDynamics(
            nn.Identity(),
            MultiEmbeddings(ACTION_CHOICES_PER_CATEGORY, ACTION_DIM, do_flatten=True),
            nn.Linear(EMBED_OBS_DIM + FLATTEN_ACTION_DIM, SCONV_DIM),
            SConv(DEPTH, SCONV_DIM, SCONV_DIM, 0.1),
            FullyConnectedFixedStdNormal(SCONV_DIM, EMBED_OBS_DIM),
        )

        policy_value = PolicyValueCommonNet(
            SmallConvNet(HEIGHT, WIDTH, CHANNELS, EMBED_OBS_DIM),
            DiscretePolicyHead(EMBED_OBS_DIM, ACTION_CHOICES_PER_CATEGORY),
            FullyConnectedValueHead(EMBED_OBS_DIM),
        )

        mwd = ModelWrappersDict(
            {
                ModelNames.IMAGE_ENCODER: ModelWrapper(image_encoder, device, True),
                ModelNames.FORWARD_DYNAMICS: ModelWrapper(
                    forward_dynamics,
                    device,
                    True,
                ),
                ModelNames.POLICY_VALUE: ModelWrapper(
                    policy_value,
                    device,
                    True,
                ),
            }
        )
        mwd.send_to_default_device()

        return mwd.inference_wrappers_dict

    @pytest.fixture
    def data_collectors(self) -> DataCollectorsDict:
        empty_buffer = RandomDataBuffer(10, [])
        return DataCollectorsDict.from_data_buffers(
            **{
                BufferNames.IMAGE: empty_buffer,
                BufferNames.PPO_TRAJECTORY: empty_buffer,
                BufferNames.FORWARD_DYNAMICS_TRAJECTORY: empty_buffer,
            }
        )

    @pytest.fixture
    def agent(self, inference_models, data_collectors) -> CuriosityImagePPOAgent:
        curiosity_agent = CuriosityImagePPOAgent(torch.zeros(DEPTH, SCONV_DIM))
        curiosity_agent.attach_data_collectors(data_collectors)
        curiosity_agent.attach_inference_models(inference_models)
        return curiosity_agent

    def test_setup_step_teardown(self, agent: CuriosityImagePPOAgent):

        observation = torch.randn(CHANNELS, HEIGHT, WIDTH)
        action = agent.setup(observation)

        assert action.shape == (len(ACTION_CHOICES_PER_CATEGORY),)

        for _ in range(10):
            action = agent.step(observation)
            assert action.shape == (len(ACTION_CHOICES_PER_CATEGORY),)

        assert agent.step_data[DataKeys.OBSERVATION].shape == observation.shape
        assert agent.step_data[DataKeys.EMBED_OBSERVATION].shape == (EMBED_OBS_DIM,)
        assert agent.step_data[DataKeys.ACTION].shape == (len(ACTION_CHOICES_PER_CATEGORY),)
        assert agent.step_data[DataKeys.ACTION_LOG_PROBABILITY].shape == (len(ACTION_CHOICES_PER_CATEGORY),)
        assert agent.step_data[DataKeys.VALUE].shape == (1,)
        assert agent.step_data[DataKeys.HIDDEN].shape == (DEPTH, SCONV_DIM)
        assert agent.step_data[DataKeys.REWARD].shape == ()
