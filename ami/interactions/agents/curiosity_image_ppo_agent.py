from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from typing_extensions import override

from ami.tensorboard_loggers import TimeIntervalLogger

from ...data.buffers.buffer_names import BufferNames
from ...data.step_data import DataKeys, StepData
from ...models.forward_dynamics import ForwardDynamcisWithActionReward, ForwardDynamics
from ...models.model_names import ModelNames
from ...models.model_wrapper import ThreadSafeInferenceWrapper
from ...models.policy_or_value_network import PolicyOrValueNetwork
from ...models.policy_value_common_net import PolicyValueCommonNet
from .base_agent import BaseAgent


class PredictionErrorReward:
    """Computes the prediction error based reward."""

    def __init__(self, scale: float = 1.0, shift: float = 0.0) -> None:
        """Constructs the class.

        Args:
            scale: Multiplies this number to reward.
            shift: Adds this number to reward.
        """
        self.scale = scale
        self.shift = shift

    def compute(self, prediction: Distribution, actual: Tensor) -> Tensor:
        """Computes the reward.

        Args:
            predicition: The prediction as the probablity distribution.
            actual: Actual value for the prediction.

        Returns:
            Tensor: The reward.
        """
        raw_reward = -prediction.log_prob(actual).mean()
        return raw_reward * self.scale + self.shift


class CuriosityImagePPOAgent(BaseAgent[Tensor, Tensor]):
    """Image input curiosity agent with ppo policy."""

    def __init__(
        self,
        initial_hidden: Tensor,
        logger: TimeIntervalLogger,
        reward: PredictionErrorReward,
        use_embed_obs_for_policy: bool = False,
    ) -> None:
        """Constructs Agent.

        Args:
            initial_hidden: Initial hidden state for the forward dynamics model.
            use_embed_obs_for_policy: Use embed observation as observation for policy input.
                Implemented for adapting the world models learning method.
        """
        super().__init__()

        self.forward_dynamics_hidden_state = initial_hidden
        self.logger = logger
        self.reward_computer = reward
        self.use_embed_obs_for_policy = use_embed_obs_for_policy

    def on_inference_models_attached(self) -> None:
        super().on_inference_models_attached()
        self.image_encoder: ThreadSafeInferenceWrapper[nn.Module] = self.get_inference_model(ModelNames.IMAGE_ENCODER)
        self.forward_dynamics: ThreadSafeInferenceWrapper[ForwardDynamics] = self.get_inference_model(
            ModelNames.FORWARD_DYNAMICS
        )
        self.policy_value: ThreadSafeInferenceWrapper[PolicyValueCommonNet] = self.get_inference_model(
            ModelNames.POLICY_VALUE
        )

    # ------ Interaction Process ------
    predicted_next_embed_observation_dist: Distribution
    forward_dynamics_hidden_state: Tensor
    step_data: StepData

    def _common_step(self, observation: Tensor, initial_step: bool = False) -> Tensor:
        """Common step procedure for agent.

        If `initial_step` is False, some procedures are skipped.
        """

        # \phi(o_t) -> z_t
        embed_obs = self.image_encoder(observation)

        if not initial_step:
            # 報酬計算は初期ステップではできないためスキップ。
            reward = self.reward_computer.compute(self.predicted_next_embed_observation_dist, embed_obs)
            self.step_data[DataKeys.REWARD] = reward  # r_{t+1}
            self.logger.log("agent/reward", reward)

            # ステップの冒頭でデータコレクトすることで前ステップのデータを収集する。
            self.data_collectors.collect(self.step_data)

        self.step_data[DataKeys.OBSERVATION] = observation  # o_t
        self.step_data[DataKeys.EMBED_OBSERVATION] = embed_obs  # z_t

        # Adaptiing World Models learning method.
        if self.use_embed_obs_for_policy:
            policy_obs = embed_obs
        else:
            policy_obs = observation

        action_dist, value = self.policy_value(policy_obs, self.forward_dynamics_hidden_state)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        self.step_data[DataKeys.ACTION] = action  # a_t
        self.step_data[DataKeys.ACTION_LOG_PROBABILITY] = action_log_prob  # log \pi(a_t | o_t)
        self.step_data[DataKeys.VALUE] = value  # v_t
        self.step_data[DataKeys.HIDDEN] = self.forward_dynamics_hidden_state  # h_t
        self.logger.log("agent/value", value)

        pred, hidden = self.forward_dynamics(embed_obs, self.forward_dynamics_hidden_state, action)
        self.predicted_next_embed_observation_dist = pred  # p(\hat{z}_{t+1} | z_t, h_t, a_t)
        self.forward_dynamics_hidden_state = hidden  # h_{t+1}

        self.logger.update()

        return action

    def setup(self) -> None:
        super().setup()
        self.step_data = StepData()
        self.initial_step = True

    def step(self, observation: Tensor) -> Tensor:
        action = self._common_step(observation, initial_step=self.initial_step)
        self.initial_step = False
        return action

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.forward_dynamics_hidden_state, path / "forward_dynamics_hidden_state.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.forward_dynamics_hidden_state = torch.load(path / "forward_dynamics_hidden_state.pt", map_location="cpu")


class CuriosityImageSeparatePolicyValueAgent(BaseAgent[Tensor, Tensor]):
    """Image input curiosity agent with separate policy and value network.

    And uses ForwardDynamicsWithActionReward
    """

    def __init__(
        self,
        initial_hidden: Tensor,
        logger: TimeIntervalLogger,
        reward: PredictionErrorReward,
        use_embed_obs_for_policy_and_value: bool = True,  # 2024/07/30 default use embed observation.
    ) -> None:
        """Constructs Agent.

        Args:
            initial_hidden: Initial hidden state for the forward dynamics model.
            use_embed_obs_for_policy_and_value: Use embed observation as observation for policy and value input.
                Implemented for adapting the world models learning method.
        """
        super().__init__()

        self.forward_dynamics_hidden_state = initial_hidden
        self.logger = logger
        self.reward_computer = reward
        self.use_embed_obs_for_policy_and_value = use_embed_obs_for_policy_and_value

    def on_inference_models_attached(self) -> None:
        super().on_inference_models_attached()
        self.image_encoder: ThreadSafeInferenceWrapper[nn.Module] = self.get_inference_model(ModelNames.IMAGE_ENCODER)
        self.forward_dynamics: ThreadSafeInferenceWrapper[ForwardDynamcisWithActionReward] = self.get_inference_model(
            ModelNames.FORWARD_DYNAMICS
        )
        self.policy_net: ThreadSafeInferenceWrapper[PolicyOrValueNetwork] = self.get_inference_model(ModelNames.POLICY)
        self.value_net: ThreadSafeInferenceWrapper[PolicyOrValueNetwork] = self.get_inference_model(ModelNames.VALUE)

    # ------ Interaction Process ------
    predicted_next_embed_observation_dist: Distribution
    predicted_reward_dist: Distribution
    forward_dynamics_hidden_state: Tensor
    step_data: StepData

    def _common_step(self, observation: Tensor, initial_step: bool = False) -> Tensor:
        """Common step procedure for agent.

        If `initial_step` is False, some procedures are skipped.
        """
        # \phi(o_t) -> z_t
        embed_obs = self.image_encoder(observation)

        if not initial_step:
            # 報酬計算は初期ステップではできないためスキップ。
            reward = self.reward_computer.compute(self.predicted_next_embed_observation_dist, embed_obs)
            self.step_data[DataKeys.REWARD] = reward  # r_{t+1}
            self.logger.log("agent/reward", reward)
            self.logger.log("agent/reward_negative_log_likelihood", -self.predicted_reward_dist.log_prob(reward).mean())

            # ステップの冒頭でデータコレクトすることで前ステップのデータを収集する。
            self.data_collectors.collect(self.step_data)

        self.step_data[DataKeys.OBSERVATION] = observation  # o_t
        self.step_data[DataKeys.EMBED_OBSERVATION] = embed_obs  # z_t

        # Adaptiing World Models learning method.
        if self.use_embed_obs_for_policy_and_value:
            pv_obs = embed_obs
        else:
            pv_obs = observation

        action_dist: Distribution = self.policy_net(pv_obs, self.forward_dynamics_hidden_state)
        value_dist: Distribution = self.value_net(pv_obs, self.forward_dynamics_hidden_state)
        action, value = action_dist.sample(), value_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        self.step_data[DataKeys.ACTION] = action  # a_t
        self.step_data[DataKeys.ACTION_LOG_PROBABILITY] = action_log_prob  # log \pi(a_t | o_t)
        self.step_data[DataKeys.VALUE] = value  # v_t
        self.step_data[DataKeys.HIDDEN] = self.forward_dynamics_hidden_state  # h_t
        self.logger.log("agent/value", value.item())

        pred_obs, _, pred_reward, hidden = self.forward_dynamics(embed_obs, self.forward_dynamics_hidden_state, action)
        self.predicted_next_embed_observation_dist = pred_obs  # p(\hat{z}_{t+1} | z_t, h_t, a_t)
        self.predicted_reward_dist = pred_reward  # p(\hat{r}_{t+1} | z_t, h_t, a_t)
        self.forward_dynamics_hidden_state = hidden  # h_{t+1}

        self.logger.update()

        return action

    def setup(self) -> None:
        super().setup()

        self.step_data = StepData()
        self.initial_step = True

    def step(self, observation: Tensor) -> Tensor:
        action = self._common_step(observation, initial_step=self.initial_step)
        self.initial_step = False
        return action

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.forward_dynamics_hidden_state, path / "forward_dynamics_hidden_state.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.forward_dynamics_hidden_state = torch.load(path / "forward_dynamics_hidden_state.pt", map_location="cpu")
