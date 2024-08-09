from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from typing_extensions import override

from ami.tensorboard_loggers import TimeIntervalLogger

from ...data.step_data import DataKeys, StepData
from ...models.forward_dynamics import ForwardDynamcisWithActionReward
from ...models.model_names import ModelNames
from ...models.model_wrapper import ThreadSafeInferenceWrapper
from ...models.policy_or_value_network import PolicyOrValueNetwork
from .base_agent import BaseAgent
from .curiosity_image_ppo_agent import PredictionErrorReward


class MultiStepImaginationCuriosityImageAgent(BaseAgent[Tensor, Tensor]):
    def __init__(
        self,
        initial_hidden: Tensor,
        logger: TimeIntervalLogger,
        max_imagination_steps: int = 1,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
    ) -> None:
        """Constructs Agent.

        Args:
            initial_hidden: Initial hidden state for the forward dynamics model.
            max_imagination_steps: Max step for imagination.
        """
        super().__init__()
        assert max_imagination_steps > 0

        self.exact_forward_dynamics_hidden_state = initial_hidden
        self.logger = logger
        self.reward_computer = PredictionErrorReward(reward_scale, reward_shift)
        self.max_imagination_steps = max_imagination_steps

    def on_inference_models_attached(self) -> None:
        super().on_inference_models_attached()
        self.image_encoder: ThreadSafeInferenceWrapper[nn.Module] = self.get_inference_model(ModelNames.IMAGE_ENCODER)
        self.forward_dynamics: ThreadSafeInferenceWrapper[ForwardDynamcisWithActionReward] = self.get_inference_model(
            ModelNames.FORWARD_DYNAMICS
        )
        self.policy_net: ThreadSafeInferenceWrapper[PolicyOrValueNetwork] = self.get_inference_model(ModelNames.POLICY)
        self.value_net: ThreadSafeInferenceWrapper[PolicyOrValueNetwork] = self.get_inference_model(ModelNames.VALUE)

    # ------ Interaction Process ------
    exact_forward_dynamics_hidden_state: Tensor
    predicted_embed_obs_dists: list[Distribution]
    predicted_embed_obses: list[Tensor]
    forward_dynamics_hidden_states: list[Tensor]
    step_data: StepData

    def _common_step(self, observation: Tensor, initial_step: bool = False) -> Tensor:
        """Common step procedure for agent.

        If `initial_step` is False, some procedures are skipped.
        """
        embed_obs = self.image_encoder(observation)

        if not initial_step:
            # 報酬計算は初期ステップではできないためスキップ。
            rewards = []
            for dist in self.predicted_embed_obs_dists:
                rewards.append(self.reward_computer.compute(dist, embed_obs))

            self.logger.log("agent/reward", rewards[0])
            for i, r in enumerate(rewards, start=1):
                self.logger.log(f"agent/reward_{i}step", r)

            # ステップの冒頭でデータコレクトすることで前ステップのデータを収集する。
            self.step_data[DataKeys.REWARD] = rewards[0]
            self.data_collectors.collect(self.step_data)

        self.step_data[DataKeys.OBSERVATION] = observation  # o_t
        self.step_data[DataKeys.EMBED_OBSERVATION] = embed_obs  # z_t

        embed_obs_list = [embed_obs, *self.predicted_embed_obses]
        hidden_list = [self.exact_forward_dynamics_hidden_state, *self.forward_dynamics_hidden_states]

        # buffer lists.
        pred_embed_obs_dist_list = []
        pred_embed_obs_list = []
        next_hidden_list = []
        action_list = []
        action_log_prob_list = []
        value_list = []

        for i in range(min(self.max_imagination_steps, len(embed_obs_list))):
            action_dist: Distribution = self.policy_net(embed_obs_list[i], hidden_list[i])
            value_dist: Distribution = self.value_net(embed_obs_list[i], hidden_list[i])
            action, value = action_dist.sample(), value_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            action_list.append(action)
            action_log_prob_list.append(action_log_prob)
            value_list.append(value)

            pred_obs_dist, _, _, hidden = self.forward_dynamics(embed_obs_list[i], hidden_list[i], action)
            pred_obs = pred_obs_dist.sample()
            pred_embed_obs_dist_list.append(pred_obs_dist)
            pred_embed_obs_list.append(pred_obs)
            next_hidden_list.append(hidden)

        self.step_data[DataKeys.ACTION] = action_list[0]  # a_t
        self.step_data[DataKeys.ACTION_LOG_PROBABILITY] = action_log_prob_list[0]  # log \pi(a_t | o_t, h_t)
        self.step_data[DataKeys.VALUE] = value_list[0]  # v_t
        self.step_data[DataKeys.HIDDEN] = self.exact_forward_dynamics_hidden_state  # h_t
        self.logger.log("agent/value", value_list[0])

        self.predicted_embed_obs_dists = pred_embed_obs_dist_list
        self.predicted_embed_obses = pred_embed_obs_list
        self.forward_dynamics_hidden_states = next_hidden_list
        self.exact_forward_dynamics_hidden_state = next_hidden_list[0]

        self.logger.update()

        return action_list[0]

    def setup(self, observation: Tensor) -> Tensor:
        super().setup(observation)
        self.step_data = StepData()
        self.predicted_embed_obs_dists = []
        self.predicted_embed_obses = []
        self.forward_dynamics_hidden_states = []

        return self._common_step(observation, initial_step=True)

    def step(self, observation: Tensor) -> Tensor:
        return self._common_step(observation, initial_step=False)

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.exact_forward_dynamics_hidden_state, path / "exact_forward_dynamics_hidden_state.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.exact_forward_dynamics_hidden_state = torch.load(
            path / "exact_forward_dynamics_hidden_state.pt", map_location="cpu"
        )
