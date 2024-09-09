from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.distributions import Distribution

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.utils import DataCollectorsDict
from ami.interactions.agents.multi_step_imagination_curiosity_agent import (
    DataKeys,
    ForwardDynamcisWithActionReward,
    ModelNames,
    MultiStepImaginationCuriosityImageAgent,
    PolicyOrValueNetwork,
    average_exponentially,
)
from ami.models.components.fully_connected_normal import FullyConnectedNormal
from ami.models.components.sioconv import SioConv
from ami.models.components.small_conv_net import SmallConvNet
from ami.models.components.small_deconv_net import SmallDeconvNet
from ami.models.policy_value_common_net import SelectObservation
from ami.models.utils import InferenceWrappersDict, ModelWrapper, ModelWrappersDict
from ami.tensorboard_loggers import TimeIntervalLogger

CHANNELS, WIDTH, HEIGHT = (3, 128, 128)
EMBED_OBS_DIM = 64
ACTION_DIM = 8
SIOCONV_DIM = 64
DEPTH = 2
HEAD = 4


class TestMultiStepImaginationCuriosityImageAgent:
    @pytest.fixture
    def inference_models(self, device) -> InferenceWrappersDict:
        image_encoder = SmallConvNet(WIDTH, HEIGHT, CHANNELS, EMBED_OBS_DIM)
        image_decoder = SmallDeconvNet(HEIGHT, WIDTH, CHANNELS, EMBED_OBS_DIM)
        forward_dynamics = ForwardDynamcisWithActionReward(
            nn.Identity(),
            nn.Identity(),
            nn.Linear(EMBED_OBS_DIM + ACTION_DIM, SIOCONV_DIM),
            SioConv(DEPTH, SIOCONV_DIM, HEAD, SIOCONV_DIM, 0.1, 16),
            FullyConnectedNormal(SIOCONV_DIM, EMBED_OBS_DIM),
            FullyConnectedNormal(SIOCONV_DIM, ACTION_DIM),
            FullyConnectedNormal(SIOCONV_DIM, 1, squeeze_feature_dim=True),
        )

        policy_net = PolicyOrValueNetwork(
            nn.Identity(),
            nn.Identity(),
            SelectObservation(),
            nn.Linear(EMBED_OBS_DIM, EMBED_OBS_DIM),
            FullyConnectedNormal(EMBED_OBS_DIM, ACTION_DIM),
        )

        value_net = PolicyOrValueNetwork(
            nn.Identity(),
            nn.Identity(),
            SelectObservation(),
            nn.Linear(EMBED_OBS_DIM, EMBED_OBS_DIM),
            FullyConnectedNormal(EMBED_OBS_DIM, 1, squeeze_feature_dim=True),
        )

        mwd = ModelWrappersDict(
            {
                ModelNames.IMAGE_ENCODER: ModelWrapper(image_encoder, device, True),
                ModelNames.IMAGE_DECODER: ModelWrapper(image_decoder, device, True),
                ModelNames.FORWARD_DYNAMICS: ModelWrapper(forward_dynamics, device, True),
                ModelNames.POLICY: ModelWrapper(policy_net, device, True),
                ModelNames.VALUE: ModelWrapper(value_net, device, True),
            }
        )
        mwd.send_to_default_device()

        return mwd.inference_wrappers_dict

    @pytest.fixture
    def data_collectors(self) -> DataCollectorsDict:
        empty_buffer = RandomDataBuffer(10, [DataKeys.OBSERVATION])
        return DataCollectorsDict.from_data_buffers(
            **{
                BufferNames.IMAGE: empty_buffer,
                BufferNames.PPO_TRAJECTORY: empty_buffer,
            }
        )

    @pytest.fixture
    def logger(self, tmp_path):
        return TimeIntervalLogger(f"{tmp_path}/tensorboard", 0)

    @pytest.fixture
    def agent(self, inference_models, data_collectors, logger, device) -> MultiStepImaginationCuriosityImageAgent:
        curiosity_agent = MultiStepImaginationCuriosityImageAgent(
            torch.zeros(DEPTH, SIOCONV_DIM, device=device),
            logger,
            max_imagination_steps=3,
            reward_average_method=partial(average_exponentially, decay=0.3),
        )
        curiosity_agent.attach_data_collectors(data_collectors)
        curiosity_agent.attach_inference_models(inference_models)
        return curiosity_agent

    def test_setup_step_teardown(self, agent: MultiStepImaginationCuriosityImageAgent):
        observation = torch.randn(CHANNELS, HEIGHT, WIDTH)
        action = agent.setup(observation)

        assert action.shape == (ACTION_DIM,)

        for _ in range(10):
            action = agent.step(observation)
            assert action.shape == (ACTION_DIM,)
            assert agent.global_step == agent.logger.global_step

        assert agent.step_data[DataKeys.OBSERVATION].shape == observation.shape
        assert agent.step_data[DataKeys.EMBED_OBSERVATION].shape == (EMBED_OBS_DIM,)
        assert agent.step_data[DataKeys.ACTION].shape == (ACTION_DIM,)
        assert agent.step_data[DataKeys.ACTION_LOG_PROBABILITY].shape == (ACTION_DIM,)
        assert agent.step_data[DataKeys.VALUE].shape == ()
        assert agent.step_data[DataKeys.HIDDEN].shape == (DEPTH, SIOCONV_DIM)
        assert agent.step_data[DataKeys.REWARD].shape == ()

        assert isinstance(agent.predicted_embed_obs_dist_imaginations, Distribution)
        assert len(agent.predicted_embed_obs_dist_imaginations.sample()) == 3
        assert len(agent.predicted_embed_obs_imaginations) == 3
        assert isinstance(agent.predicted_embed_obs_imaginations, torch.Tensor)
        assert len(agent.forward_dynamics_hidden_state_imaginations) == 3
        assert isinstance(agent.forward_dynamics_hidden_state_imaginations, torch.Tensor)

    def test_save_and_load_state(self, agent: MultiStepImaginationCuriosityImageAgent, tmp_path):
        agent_path = tmp_path / "agent"
        agent.save_state(agent_path)
        assert (agent_path / "exact_forward_dynamics_hidden_state.pt").exists()

        hidden = agent.exact_forward_dynamics_hidden_state.clone()
        agent.exact_forward_dynamics_hidden_state = torch.randn_like(hidden)
        assert not torch.equal(agent.exact_forward_dynamics_hidden_state, hidden)

        agent.load_state(agent_path)
        assert torch.equal(agent.exact_forward_dynamics_hidden_state, hidden)


def test_average_exponentially():
    rewards = torch.ones(3)

    out = average_exponentially(rewards, 0.1)
    assert out.ndim == 0
    assert out == pytest.approx(1)

    rewards = torch.Tensor([1, 10, 100])
    assert average_exponentially(rewards, 0.1) == pytest.approx(3 / 1.11)

    # Assert decay < 0, or decay >= 1
    with pytest.raises(AssertionError):
        average_exponentially(rewards, -1)

    with pytest.raises(AssertionError):
        average_exponentially(rewards, 1)
