from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.distributions import Distribution
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.step_data import DataKeys, StepData
from ami.models.forward_dynamics import (
    ForwardDynamcisWithActionReward,
    ForwardDynamics,
    PrimitiveForwardDynamics,
)
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ThreadSafeInferenceWrapper
from ami.models.policy_value_common_net import (
    PolicyValueCommonNet,
    PrimitivePolicyValueCommonNet,
    TemporalPolicyValueCommonNet,
)
from ami.tensorboard_loggers import TimeIntervalLogger

from .base_agent import BaseAgent
from .utils import PolicyValueCommonProxy


class CuriosityAgent(BaseAgent[Tensor, Tensor]):
    """A reinforcement learning agent that uses curiosity-driven exploration
    through forward dynamics prediction.

    This agent implements curiosity-driven exploration by predicting
    future observations and using prediction errors as intrinsic
    rewards. It maintains a forward dynamics model to predict future
    states and a policy-value network for action selection.
    """

    @override
    def __init__(
        self,
        initial_hidden: Tensor,
        logger: TimeIntervalLogger,
        max_imagination_steps: int = 1,
        reward_average_method: Callable[[Tensor], Tensor] = torch.mean,
    ) -> None:
        """Initializes the CuriosityAgent.

        Args:
            initial_hidden (Tensor): Initial hidden state tensor for the forward dynamics model.
            logger (TimeIntervalLogger): Logger instance for recording metrics and rewards.
            max_imagination_steps (int, optional): Maximum number of steps to imagine into the future. Must be >= 1. Defaults to 1.
            reward_average_method (Callable[[Tensor], Tensor], optional): Function to average rewards across imagination steps.
                Takes a tensor of rewards (imagination_steps,) and returns a scalar reward. Defaults to torch.mean.
        """
        super().__init__()
        if max_imagination_steps < 1:
            raise ValueError(f"`max_imagination_steps` must be >= 1! Your input: {max_imagination_steps}")

        self.head_forward_dynamics_hidden_state = initial_hidden
        self.logger = logger
        self.max_imagination_steps = max_imagination_steps
        self.reward_average_method = reward_average_method

    @override
    def on_inference_models_attached(self) -> None:
        super().on_inference_models_attached()
        self.forward_dynamics: ThreadSafeInferenceWrapper[ForwardDynamcisWithActionReward] = self.get_inference_model(
            ModelNames.FORWARD_DYNAMICS
        )
        self.policy_value = self._get_policy_value_net()

    def _get_policy_value_net(self) -> ThreadSafeInferenceWrapper[PolicyValueCommonNet] | PolicyValueCommonProxy:
        if self.check_model_exists(ModelNames.POLICY_VALUE):
            return self.get_inference_model(ModelNames.POLICY_VALUE)
        else:
            policy = self.get_inference_model(ModelNames.POLICY)
            value = self.get_inference_model(ModelNames.VALUE)
            return PolicyValueCommonProxy(policy, value)

    @override
    def on_data_collectors_attached(self) -> None:
        super().on_data_collectors_attached()
        self.forward_dynamics_collector = self.get_data_collector(
            BufferNames.FORWARD_DYNAMICS_TRAJECTORY,
        )
        self.policy_collector = self.get_data_collector(
            BufferNames.PPO_TRAJECTORY,
        )

    # ###### INTERACTION PROCESS ########

    head_forward_dynamics_hidden_state: Tensor  # (depth, dim)
    obs_dist_imaginations: Distribution  # (imaginations, dim)
    obs_imaginations: Tensor  # (imaginations, dim)
    forward_dynamics_hidden_imaginations: Tensor  # (imaginations, depth, dim)
    step_data: StepData

    @override
    def setup(self) -> None:
        super().setup()
        self.step_data = StepData()
        self.forward_dynamics_hidden_imaginations = torch.empty(0).type_as(self.head_forward_dynamics_hidden_state)
        self.obs_imaginations = torch.empty(0, device=self.forward_dynamics_hidden_imaginations.device)
        self.initial_step = True

    @override
    def step(self, observation: Tensor) -> Tensor:
        action = self._common_step(observation, self.initial_step)
        self.initial_step = False
        return action

    def _common_step(self, observation: Tensor, initial_step: bool) -> Tensor:
        """Executes the common step procedure for the curiosity-driven agent.

        Args:
            observation (Tensor): Current observation from the environment
            initial_step (bool): Whether this is the first step in an episode.
                When True, skips reward calculation as there are no previous predictions.

        Returns:
            Tensor: Selected action to be executed in the environment
        """
        observation = observation.type_as(self.obs_imaginations)  # convert type and send to device

        if not initial_step:
            target_obses = observation.expand_as(self.obs_imaginations)
            reward_imaginations = -self.obs_dist_imaginations.log_prob(target_obses).flatten(1).mean(-1)

            reward = self.reward_average_method(reward_imaginations)
            self.logger.log("curiosity_agent/reward", reward)

            self.step_data[DataKeys.REWARD] = reward
            self.forward_dynamics_collector.collect(self.step_data)
            self.policy_collector.collect(self.step_data)

        obs_imaginations = torch.cat([observation[None], self.obs_imaginations])[
            : self.max_imagination_steps
        ]  # (imaginations, dim)
        hidden_imaginations = torch.cat(
            [self.head_forward_dynamics_hidden_state[None], self.forward_dynamics_hidden_imaginations]
        )[
            : self.max_imagination_steps
        ]  # (imaginations, depth, dim)

        action_dist: Distribution
        value: Tensor
        action_dist, value = self.policy_value(obs_imaginations[0], hidden_imaginations[0])
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        obs_dist_imaginations, _, _, hidden_imaginations = self.forward_dynamics(
            obs_imaginations, hidden_imaginations, action.expand((len(obs_imaginations), *action.shape))
        )
        obs_imaginations = obs_dist_imaginations.sample()

        self.step_data[DataKeys.OBSERVATION] = observation.cpu()
        self.step_data[DataKeys.ACTION] = action
        self.step_data[DataKeys.ACTION_LOG_PROBABILITY] = action_log_prob
        self.step_data[DataKeys.VALUE] = value
        self.step_data[DataKeys.HIDDEN] = self.head_forward_dynamics_hidden_state
        self.logger.log("curiosity_agent/value", value)

        self.obs_dist_imaginations = obs_dist_imaginations
        self.obs_imaginations = obs_imaginations
        self.forward_dynamics_hidden_imaginations = hidden_imaginations
        self.head_forward_dynamics_hidden_state = hidden_imaginations[0]

        self.logger.update()
        return action

    # ###### State saving ######
    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.head_forward_dynamics_hidden_state, path / "head_forward_dynamics_hidden_state.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.head_forward_dynamics_hidden_state = torch.load(
            path / "head_forward_dynamics_hidden_state.pt",
            map_location=self.head_forward_dynamics_hidden_state.device,
        )


class PrimitiveCuriosityAgent(BaseAgent[Tensor, Tensor]):
    """Primitive Curiosity Agent Implementation."""

    @override
    def __init__(
        self,
        logger: TimeIntervalLogger,
    ) -> None:
        """Initializes the PrimitiveCuriosityAgent.

        Args:
            logger: Logger instance for recording metrics and rewards.
        """
        super().__init__()
        self.logger = logger

    @override
    def on_inference_models_attached(self) -> None:
        super().on_inference_models_attached()
        self.forward_dynamics: ThreadSafeInferenceWrapper[PrimitiveForwardDynamics] = self.get_inference_model(
            ModelNames.FORWARD_DYNAMICS
        )
        self.policy_value: ThreadSafeInferenceWrapper[PrimitivePolicyValueCommonNet] = self.get_inference_model(
            ModelNames.POLICY_VALUE
        )

    @override
    def on_data_collectors_attached(self) -> None:
        super().on_data_collectors_attached()
        self.forward_dynamics_collector = self.get_data_collector(
            BufferNames.FORWARD_DYNAMICS_TRAJECTORY,
        )
        self.policy_collector = self.get_data_collector(
            BufferNames.PPO_TRAJECTORY,
        )

    # ###### INTERACTION PROCESS ########

    step_data: StepData
    predicted_obs_dist: Distribution
    predicted_obs_device: torch.device

    @override
    def setup(self) -> None:
        super().setup()
        self.step_data = StepData()
        self.initial_step = True

    @override
    def step(self, observation: Tensor) -> Tensor:
        action = self._common_step(observation, self.initial_step)
        self.initial_step = False
        return action

    def _common_step(self, observation: Tensor, initial_step: bool) -> Tensor:
        """Executes the common step procedure for the curiosity-driven agent.

        Args:
            observation (Tensor): Current observation from the environment
            initial_step (bool): Whether this is the first step in an episode.
                When True, skips reward calculation as there are no previous predictions.

        Returns:
            Tensor: Selected action to be executed in the environment
        """

        if not initial_step:
            observation = observation.to(self.predicted_obs_device)  # convert type and send to device
            reward = -self.predicted_obs_dist.log_prob(observation).mean()
            self.logger.log("curiosity_agent/reward", reward)

            self.step_data[DataKeys.REWARD] = reward
            self.forward_dynamics_collector.collect(self.step_data)
            self.policy_collector.collect(self.step_data)

        action_dist: Distribution
        value: Tensor
        action_dist, value = self.policy_value(observation)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        predicted_obs_dist = self.forward_dynamics(observation, action)

        self.step_data[DataKeys.OBSERVATION] = observation.cpu()
        self.step_data[DataKeys.ACTION] = action
        self.step_data[DataKeys.ACTION_LOG_PROBABILITY] = action_log_prob
        self.step_data[DataKeys.VALUE] = value
        self.logger.log("curiosity_agent/value", value)

        self.predicted_obs_dist = predicted_obs_dist
        self.predicted_obs_device = predicted_obs_dist.sample().device

        self.logger.update()
        return action

    # ###### State saving ######
    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.logger.state_dict(), path / "logger.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.logger.load_state_dict(torch.load(path / "logger.pt"))


class IsolatedHiddenCuriosityAgent(BaseAgent[Tensor, Tensor]):
    """A reinforcement learning agent that uses curiosity-driven exploration
    through forward dynamics prediction.

    This agent implements curiosity-driven exploration by predicting
    future observations and using prediction errors as intrinsic
    rewards. It maintains a forward dynamics model to predict future
    states and a policy-value network for action selection.
    """

    @override
    def __init__(
        self,
        initial_forward_dynamics_hidden: Tensor,
        initial_policy_hidden: Tensor,
        logger: TimeIntervalLogger,
        max_imagination_steps: int = 1,
        reward_average_method: Callable[[Tensor], Tensor] = torch.mean,
    ) -> None:
        """Initializes the CuriosityAgent.

        Args:
            initial_hidden (Tensor): Initial hidden state tensor for the forward dynamics model.
            logger (TimeIntervalLogger): Logger instance for recording metrics and rewards.
            max_imagination_steps (int, optional): Maximum number of steps to imagine into the future. Must be >= 1. Defaults to 1.
            reward_average_method (Callable[[Tensor], Tensor], optional): Function to average rewards across imagination steps.
                Takes a tensor of rewards (imagination_steps,) and returns a scalar reward. Defaults to torch.mean.
        """
        super().__init__()
        if max_imagination_steps < 1:
            raise ValueError(f"`max_imagination_steps` must be >= 1! Your input: {max_imagination_steps}")

        self.head_forward_dynamics_hidden_state = initial_forward_dynamics_hidden
        self.policy_hidden_state = initial_policy_hidden
        self.logger = logger
        self.max_imagination_steps = max_imagination_steps
        self.reward_average_method = reward_average_method

    @override
    def on_inference_models_attached(self) -> None:
        super().on_inference_models_attached()
        self.forward_dynamics: ThreadSafeInferenceWrapper[ForwardDynamics] = self.get_inference_model(
            ModelNames.FORWARD_DYNAMICS
        )
        self.policy_value: ThreadSafeInferenceWrapper[TemporalPolicyValueCommonNet] = self.get_inference_model(
            ModelNames.POLICY_VALUE
        )

    @override
    def on_data_collectors_attached(self) -> None:
        super().on_data_collectors_attached()
        self.forward_dynamics_collector = self.get_data_collector(
            BufferNames.FORWARD_DYNAMICS_TRAJECTORY,
        )
        self.policy_collector = self.get_data_collector(
            BufferNames.PPO_TRAJECTORY,
        )

    # ###### INTERACTION PROCESS ########

    head_forward_dynamics_hidden_state: Tensor  # (depth, dim)
    policy_hidden_state: Tensor  # (depth, dim)
    obs_dist_imaginations: Distribution  # (imaginations, dim)
    obs_imaginations: Tensor  # (imaginations, dim)
    forward_dynamics_hidden_imaginations: Tensor  # (imaginations, depth, dim)
    step_data_fd: StepData
    step_data_policy: StepData

    @override
    def setup(self) -> None:
        super().setup()
        self.step_data_fd, self.step_data_policy = StepData(), StepData()
        self.forward_dynamics_hidden_imaginations = torch.empty(0).type_as(self.head_forward_dynamics_hidden_state)
        self.obs_imaginations = torch.empty(0, device=self.forward_dynamics_hidden_imaginations.device)
        self.initial_step = True

    @override
    def step(self, observation: Tensor) -> Tensor:
        action = self._common_step(observation, self.initial_step)
        self.initial_step = False
        return action

    def _common_step(self, observation: Tensor, initial_step: bool) -> Tensor:
        """Executes the common step procedure for the curiosity-driven agent.

        Args:
            observation (Tensor): Current observation from the environment
            initial_step (bool): Whether this is the first step in an episode.
                When True, skips reward calculation as there are no previous predictions.

        Returns:
            Tensor: Selected action to be executed in the environment
        """
        observation = observation.type_as(self.obs_imaginations)  # convert type and send to device

        if not initial_step:
            target_obses = observation.expand_as(self.obs_imaginations)
            reward_imaginations = -self.obs_dist_imaginations.log_prob(target_obses).flatten(1).mean(-1)

            reward = self.reward_average_method(reward_imaginations)
            self.logger.log("curiosity_agent/reward", reward)

            self.step_data_policy[DataKeys.REWARD] = reward
            self.policy_collector.collect(self.step_data_policy)

        obs_imaginations = torch.cat([observation[None], self.obs_imaginations])[
            : self.max_imagination_steps
        ]  # (imaginations, dim)
        hidden_imaginations = torch.cat(
            [self.head_forward_dynamics_hidden_state[None], self.forward_dynamics_hidden_imaginations]
        )[
            : self.max_imagination_steps
        ]  # (imaginations, depth, dim)

        self.step_data_policy[DataKeys.HIDDEN] = self.policy_hidden_state
        action_dist: Distribution
        value: Tensor
        action_dist, value, policy_hidden_state = self.policy_value(obs_imaginations[0], hidden_imaginations[0])
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        self.step_data_fd[DataKeys.HIDDEN] = self.head_forward_dynamics_hidden_state

        obs_dist_imaginations, hidden_imaginations = self.forward_dynamics(
            obs_imaginations, hidden_imaginations, action.expand((len(obs_imaginations), *action.shape))
        )
        obs_imaginations = obs_dist_imaginations.sample()

        self.step_data_fd[DataKeys.OBSERVATION] = self.step_data_policy[DataKeys.OBSERVATION] = observation.cpu()
        self.step_data_fd[DataKeys.ACTION] = self.step_data_policy[DataKeys.ACTION] = action
        self.forward_dynamics_collector.collect(self.step_data_fd)

        self.step_data_policy[DataKeys.ACTION_LOG_PROBABILITY] = action_log_prob
        self.step_data_policy[DataKeys.VALUE] = value
        self.logger.log("curiosity_agent/value", value)

        self.obs_dist_imaginations = obs_dist_imaginations
        self.obs_imaginations = obs_imaginations
        self.forward_dynamics_hidden_imaginations = hidden_imaginations
        self.head_forward_dynamics_hidden_state = hidden_imaginations[0]
        self.policy_hidden_state = policy_hidden_state

        self.logger.update()
        return action

    # ###### State saving ######
    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.head_forward_dynamics_hidden_state, path / "head_forward_dynamics_hidden_state.pt")
        torch.save(self.policy_hidden_state, path / "policy_hidden_state.pt")
        torch.save(self.logger.state_dict(), path / "logger.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.head_forward_dynamics_hidden_state = torch.load(
            path / "head_forward_dynamics_hidden_state.pt",
            map_location=self.head_forward_dynamics_hidden_state.device,
        )
        self.policy_hidden_state = torch.load(
            path / "policy_hidden_state.pt", map_location=self.policy_hidden_state.device
        )
        self.logger.load_state_dict(torch.load(path / "logger.pt"))
