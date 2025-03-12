import pytest
import torch
import torch.nn as nn
from torch.distributions import Distribution

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.step_data import DataKeys
from ami.data.utils import DataCollectorsDict
from ami.interactions.agents.curiosity_agent import (
    CuriosityAgent,
    IsolatedHiddenCuriosityAgent,
    PrimitiveCuriosityAgent,
)
from ami.models.components.fully_connected_normal import FullyConnectedNormal
from ami.models.components.fully_connected_value_head import FullyConnectedValueHead
from ami.models.components.sioconv import SioConv
from ami.models.forward_dynamics import (
    ForwardDynamcisWithActionReward,
    ForwardDynamics,
    PrimitiveForwardDynamics,
)
from ami.models.model_names import ModelNames
from ami.models.policy_or_value_network import PolicyOrValueNetwork
from ami.models.policy_value_common_net import (
    PolicyValueCommonNet,
    PrimitivePolicyValueCommonNet,
    SelectObservation,
    TemporalPolicyValueCommonNet,
)
from ami.models.utils import InferenceWrappersDict, ModelWrapper, ModelWrappersDict
from ami.tensorboard_loggers import TimeIntervalLogger

# Test constants
OBSERVATION_DIM = 64
ACTION_DIM = 8
HIDDEN_DIM = 64
DEPTH = 2
HEAD = 4


class TestCuriosityAgent:
    @pytest.fixture
    def models(self, device) -> ModelWrappersDict:
        # Create forward dynamics model
        forward_dynamics = ForwardDynamcisWithActionReward(
            nn.Identity(),
            nn.Identity(),
            nn.Linear(OBSERVATION_DIM + ACTION_DIM, HIDDEN_DIM),
            SioConv(DEPTH, HIDDEN_DIM, HEAD, HIDDEN_DIM, 0.1, 16),
            FullyConnectedNormal(HIDDEN_DIM, OBSERVATION_DIM),
            FullyConnectedNormal(HIDDEN_DIM, ACTION_DIM),
            FullyConnectedNormal(HIDDEN_DIM, 1, squeeze_feature_dim=True),
        )

        # Create policy network
        policy_net = PolicyOrValueNetwork(
            nn.Identity(),
            nn.Identity(),
            SelectObservation(),
            nn.Linear(OBSERVATION_DIM, HIDDEN_DIM),
            FullyConnectedNormal(HIDDEN_DIM, ACTION_DIM),
        )

        # Create value network
        value_net = PolicyOrValueNetwork(
            nn.Identity(),
            nn.Identity(),
            SelectObservation(),
            nn.Linear(OBSERVATION_DIM, HIDDEN_DIM),
            FullyConnectedNormal(HIDDEN_DIM, 1, squeeze_feature_dim=True),
        )

        return {
            ModelNames.FORWARD_DYNAMICS: ModelWrapper(forward_dynamics, device, True),
            ModelNames.POLICY: ModelWrapper(policy_net, device, True),
            ModelNames.VALUE: ModelWrapper(value_net, device, True),
        }

    @pytest.fixture
    def inference_models(self, models) -> InferenceWrappersDict:
        mwd = ModelWrappersDict(models)
        mwd.send_to_default_device()

        return mwd.inference_wrappers_dict

    @pytest.fixture
    def inference_models_policy_value_common(self, models) -> InferenceWrappersDict:
        del models[ModelNames.POLICY]
        del models[ModelNames.VALUE]
        policy_value = PolicyValueCommonNet(
            nn.Identity(),
            nn.Identity(),
            SelectObservation(),
            nn.Linear(OBSERVATION_DIM, HIDDEN_DIM),
            FullyConnectedNormal(HIDDEN_DIM, ACTION_DIM),
            FullyConnectedValueHead(HIDDEN_DIM, squeeze_value_dim=True),
        )
        models[ModelNames.POLICY_VALUE] = ModelWrapper(policy_value)
        mwd = ModelWrappersDict(models)
        mwd.send_to_default_device()
        return mwd.inference_wrappers_dict

    @pytest.fixture
    def data_collectors(self) -> DataCollectorsDict:
        empty_buffer = RandomDataBuffer(10, [DataKeys.OBSERVATION])
        return DataCollectorsDict.from_data_buffers(
            **{
                BufferNames.FORWARD_DYNAMICS_TRAJECTORY: empty_buffer,
                BufferNames.PPO_TRAJECTORY: empty_buffer,
            }
        )

    @pytest.fixture
    def logger(self, tmp_path):
        return TimeIntervalLogger(f"{tmp_path}/tensorboard", 0)

    @pytest.fixture
    def agent(self, inference_models, data_collectors, logger, device) -> CuriosityAgent:
        curiosity_agent = CuriosityAgent(
            initial_hidden=torch.zeros(DEPTH, HIDDEN_DIM, device=device),
            logger=logger,
            max_imagination_steps=3,
        )
        curiosity_agent.attach_data_collectors(data_collectors)
        curiosity_agent.attach_inference_models(inference_models)
        return curiosity_agent

    def test_setup_step_teardown(self, agent: CuriosityAgent):
        """Test the main interaction loop of the agent."""
        observation = torch.randn(OBSERVATION_DIM)
        agent.setup()

        assert agent.initial_step
        for _ in range(10):
            action = agent.step(observation)
            assert not agent.initial_step
            assert action.shape == (ACTION_DIM,)
            assert isinstance(action, torch.Tensor)

        # Check step data shapes
        assert agent.step_data[DataKeys.OBSERVATION].shape == (OBSERVATION_DIM,)
        assert agent.step_data[DataKeys.ACTION].shape == (ACTION_DIM,)
        assert agent.step_data[DataKeys.ACTION_LOG_PROBABILITY].shape == (ACTION_DIM,)
        assert agent.step_data[DataKeys.VALUE].shape == ()
        assert agent.step_data[DataKeys.HIDDEN].shape == (DEPTH, HIDDEN_DIM)
        assert agent.step_data[DataKeys.REWARD].shape == ()

        # Check imagination states
        assert isinstance(agent.obs_dist_imaginations, Distribution)
        assert len(agent.obs_imaginations) == 3
        assert isinstance(agent.obs_imaginations, torch.Tensor)
        assert len(agent.forward_dynamics_hidden_imaginations) == 3
        assert isinstance(agent.forward_dynamics_hidden_imaginations, torch.Tensor)

    def test_save_and_load_state(self, agent: CuriosityAgent, tmp_path):
        """Test state saving and loading functionality."""
        agent_path = tmp_path / "agent"
        agent.save_state(agent_path)
        assert (agent_path / "head_forward_dynamics_hidden_state.pt").exists()

        # Modify state and verify it's different
        hidden = agent.head_forward_dynamics_hidden_state.clone()
        agent.head_forward_dynamics_hidden_state = torch.randn_like(hidden)
        assert not torch.equal(agent.head_forward_dynamics_hidden_state, hidden)

        # Load state and verify it's restored
        agent.load_state(agent_path)
        assert torch.equal(agent.head_forward_dynamics_hidden_state, hidden)

    def test_initialization_with_invalid_steps(self):
        """Test that agent raises error for invalid max_imagination_steps."""
        with pytest.raises(ValueError):
            CuriosityAgent(initial_hidden=torch.zeros(DEPTH, HIDDEN_DIM), logger=None, max_imagination_steps=0)

    def test_policy_value_common_net(self, inference_models_policy_value_common, data_collectors, logger, device):
        """Test agent with combined policy-value network."""

        # Create agent with combined network
        agent = CuriosityAgent(
            initial_hidden=torch.zeros(DEPTH, HIDDEN_DIM, device=device),
            logger=logger,
            max_imagination_steps=3,
        )
        agent.attach_data_collectors(data_collectors)
        agent.attach_inference_models(inference_models_policy_value_common)

        # Test that agent works with combined network
        observation = torch.randn(OBSERVATION_DIM)
        agent.setup()
        action = agent.step(observation)
        assert action.shape == (ACTION_DIM,)


class TestPrimitiveCuriosityAgent:
    @pytest.fixture
    def models(self, device) -> ModelWrappersDict:
        # Create forward dynamics model
        forward_dynamics = PrimitiveForwardDynamics(
            nn.Identity(),
            nn.Identity(),
            nn.Linear(OBSERVATION_DIM + ACTION_DIM, HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            FullyConnectedNormal(HIDDEN_DIM, OBSERVATION_DIM),
        )
        policy_value = PrimitivePolicyValueCommonNet(
            nn.Linear(OBSERVATION_DIM, HIDDEN_DIM),
            nn.Identity(),
            FullyConnectedNormal(HIDDEN_DIM, ACTION_DIM),
            FullyConnectedValueHead(HIDDEN_DIM, squeeze_value_dim=True),
        )

        return {
            ModelNames.FORWARD_DYNAMICS: ModelWrapper(forward_dynamics, device, True),
            ModelNames.POLICY_VALUE: ModelWrapper(policy_value, device, True),
        }

    @pytest.fixture
    def inference_models(self, models) -> InferenceWrappersDict:
        mwd = ModelWrappersDict(models)
        mwd.send_to_default_device()

        return mwd.inference_wrappers_dict

    @pytest.fixture
    def data_collectors(self) -> DataCollectorsDict:
        empty_buffer = RandomDataBuffer(10, [DataKeys.OBSERVATION])
        return DataCollectorsDict.from_data_buffers(
            **{
                BufferNames.FORWARD_DYNAMICS_TRAJECTORY: empty_buffer,
                BufferNames.PPO_TRAJECTORY: empty_buffer,
            }
        )

    @pytest.fixture
    def logger(self, tmp_path):
        return TimeIntervalLogger(f"{tmp_path}/tensorboard", 0)

    @pytest.fixture
    def agent(self, inference_models, data_collectors, logger) -> PrimitiveCuriosityAgent:
        curiosity_agent = PrimitiveCuriosityAgent(
            logger=logger,
        )
        curiosity_agent.attach_data_collectors(data_collectors)
        curiosity_agent.attach_inference_models(inference_models)
        return curiosity_agent

    def test_setup_step_teardown(self, agent: PrimitiveCuriosityAgent, device):
        """Test the main interaction loop of the agent."""
        observation = torch.randn(OBSERVATION_DIM)
        agent.setup()

        assert agent.initial_step
        for _ in range(10):
            action = agent.step(observation)
            assert not agent.initial_step
            assert action.shape == (ACTION_DIM,)
            assert isinstance(action, torch.Tensor)

        # Check step data shapes
        assert agent.step_data[DataKeys.OBSERVATION].shape == (OBSERVATION_DIM,)
        assert agent.step_data[DataKeys.ACTION].shape == (ACTION_DIM,)
        assert agent.step_data[DataKeys.ACTION_LOG_PROBABILITY].shape == (ACTION_DIM,)
        assert agent.step_data[DataKeys.VALUE].shape == ()
        assert agent.step_data[DataKeys.REWARD].shape == ()

        # Check imagination states
        assert isinstance(agent.predicted_obs_dist, Distribution)
        assert agent.predicted_obs_device == device

    def test_save_and_load_state(self, agent: CuriosityAgent, tmp_path):
        """Test state saving and loading functionality."""
        agent_path = tmp_path / "agent"
        agent.save_state(agent_path)
        assert (agent_path / "logger.pt").exists()

        # Modify state and verify it's different
        logger_state = agent.logger.state_dict()
        agent.logger.global_step = -1
        assert agent.logger.state_dict() != logger_state

        # Load state and verify it's restored
        agent.load_state(agent_path)
        assert agent.logger.state_dict() == logger_state


class TestIsolatedHiddenCuriosityAgent:
    @pytest.fixture
    def models(self, device) -> ModelWrappersDict:
        # Create forward dynamics model
        forward_dynamics = ForwardDynamics(
            nn.Identity(),
            nn.Identity(),
            nn.Linear(OBSERVATION_DIM + ACTION_DIM, HIDDEN_DIM),
            SioConv(DEPTH, HIDDEN_DIM, HEAD, HIDDEN_DIM, 0.1, 16),
            FullyConnectedNormal(HIDDEN_DIM, OBSERVATION_DIM),
        )

        policy_value = TemporalPolicyValueCommonNet(
            observation_flatten=nn.Linear(OBSERVATION_DIM, HIDDEN_DIM),
            core_model=SioConv(DEPTH, HIDDEN_DIM, HEAD, HIDDEN_DIM, 0.1, 16),
            policy_head=FullyConnectedNormal(HIDDEN_DIM, ACTION_DIM),
            value_head=FullyConnectedValueHead(HIDDEN_DIM, squeeze_value_dim=True),
        )

        return {
            ModelNames.FORWARD_DYNAMICS: ModelWrapper(forward_dynamics, device, True),
            ModelNames.POLICY_VALUE: ModelWrapper(policy_value, device, True),
        }

    @pytest.fixture
    def inference_models(self, models) -> InferenceWrappersDict:
        mwd = ModelWrappersDict(models)
        mwd.send_to_default_device()

        return mwd.inference_wrappers_dict

    @pytest.fixture
    def data_collectors(self) -> DataCollectorsDict:
        empty_buffer = RandomDataBuffer(10, [DataKeys.OBSERVATION])
        return DataCollectorsDict.from_data_buffers(
            **{
                BufferNames.FORWARD_DYNAMICS_TRAJECTORY: empty_buffer,
                BufferNames.PPO_TRAJECTORY: empty_buffer,
            }
        )

    @pytest.fixture
    def logger(self, tmp_path):
        return TimeIntervalLogger(f"{tmp_path}/tensorboard", 0)

    @pytest.fixture
    def agent(self, inference_models, data_collectors, logger, device) -> IsolatedHiddenCuriosityAgent:
        curiosity_agent = IsolatedHiddenCuriosityAgent(
            initial_forward_dynamics_hidden=torch.zeros(DEPTH, HIDDEN_DIM, device=device),
            initial_policy_hidden=torch.zeros(DEPTH, HIDDEN_DIM, device=device),
            logger=logger,
            max_imagination_steps=3,
        )
        curiosity_agent.attach_data_collectors(data_collectors)
        curiosity_agent.attach_inference_models(inference_models)
        return curiosity_agent

    def test_setup_step_teardown(self, agent: IsolatedHiddenCuriosityAgent):
        """Test the main interaction loop of the agent."""
        observation = torch.randn(OBSERVATION_DIM)
        agent.setup()

        assert agent.initial_step
        for _ in range(10):
            action = agent.step(observation)
            assert not agent.initial_step
            assert action.shape == (ACTION_DIM,)
            assert isinstance(action, torch.Tensor)

        # Check step data shapes
        assert agent.step_data_fd[DataKeys.OBSERVATION].shape == (OBSERVATION_DIM,)
        assert agent.step_data_fd[DataKeys.ACTION].shape == (ACTION_DIM,)
        assert agent.step_data_fd[DataKeys.HIDDEN].shape == (DEPTH, HIDDEN_DIM)

        assert agent.step_data_policy[DataKeys.OBSERVATION].shape == (OBSERVATION_DIM,)
        assert agent.step_data_policy[DataKeys.ACTION].shape == (ACTION_DIM,)
        assert agent.step_data_policy[DataKeys.ACTION_LOG_PROBABILITY].shape == (ACTION_DIM,)
        assert agent.step_data_policy[DataKeys.VALUE].shape == ()
        assert agent.step_data_policy[DataKeys.HIDDEN].shape == (DEPTH, HIDDEN_DIM)
        assert agent.step_data_policy[DataKeys.REWARD].shape == ()

        assert not torch.allclose(agent.step_data_fd[DataKeys.HIDDEN], agent.step_data_policy[DataKeys.HIDDEN])

        # Check imagination states
        assert isinstance(agent.obs_dist_imaginations, Distribution)
        assert len(agent.obs_imaginations) == 3
        assert isinstance(agent.obs_imaginations, torch.Tensor)
        assert len(agent.forward_dynamics_hidden_imaginations) == 3
        assert isinstance(agent.forward_dynamics_hidden_imaginations, torch.Tensor)

    def test_save_and_load_state(self, agent: IsolatedHiddenCuriosityAgent, tmp_path):
        """Test state saving and loading functionality."""
        agent_path = tmp_path / "agent"
        agent.save_state(agent_path)
        assert (agent_path / "head_forward_dynamics_hidden_state.pt").exists()
        assert (agent_path / "policy_hidden_state.pt").exists()

        # Modify state and verify it's different
        fd_hidden = agent.head_forward_dynamics_hidden_state.clone()
        policy_hidden = agent.policy_hidden_state.clone()

        agent.head_forward_dynamics_hidden_state = torch.randn_like(fd_hidden)
        agent.policy_hidden_state = torch.randn_like(policy_hidden)
        assert not torch.equal(agent.head_forward_dynamics_hidden_state, fd_hidden)
        assert not torch.equal(agent.policy_hidden_state, policy_hidden)

        # Load state and verify it's restored
        agent.load_state(agent_path)
        assert torch.equal(agent.head_forward_dynamics_hidden_state, fd_hidden)
        assert torch.equal(agent.policy_hidden_state, policy_hidden)

    def test_initialization_with_invalid_steps(self, logger):
        """Test that agent raises error for invalid max_imagination_steps."""
        with pytest.raises(ValueError):
            IsolatedHiddenCuriosityAgent(
                initial_forward_dynamics_hidden=torch.zeros(DEPTH, HIDDEN_DIM),
                initial_policy_hidden=torch.zeros(DEPTH, HIDDEN_DIM),
                logger=logger,
                max_imagination_steps=0,
            )
