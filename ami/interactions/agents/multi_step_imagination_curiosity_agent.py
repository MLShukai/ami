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
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift
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
    exact_forward_dynamics_hidden_state: Tensor  # (depth, dim)
    predicted_embed_obs_dists: Distribution  # (batch, dim)
    predicted_embed_obses: Tensor  # (batch, dim)
    forward_dynamics_hidden_states: Tensor  # (batch, depth, dim)
    step_data: StepData

    def _common_step(self, observation: Tensor, initial_step: bool = False) -> Tensor:
        """Common step procedure for agent.

        If `initial_step` is False, some procedures are skipped.
        """
        embed_obs: Tensor = self.image_encoder(observation)

        if not initial_step:
            # 報酬計算は初期ステップではできないためスキップ。
            embed_obs = embed_obs.type_as(self.predicted_embed_obses)
            target_obses = embed_obs.expand_as(self.predicted_embed_obses)
            reward_batch = (
                -self.predicted_embed_obs_dists.log_prob(target_obses).flatten(1).mean(-1) * self.reward_scale
                + self.reward_shift
            )

            self.logger.log("agent/reward", reward_batch[0])
            for i, r in enumerate(reward_batch, start=1):
                self.logger.log(f"agent/reward_{i}step", r)

            # ステップの冒頭でデータコレクトすることで前ステップのデータを収集する。
            self.step_data[DataKeys.REWARD] = reward_batch[0]
            self.data_collectors.collect(self.step_data)

        self.step_data[DataKeys.OBSERVATION] = observation  # o_t
        self.step_data[DataKeys.EMBED_OBSERVATION] = embed_obs  # z_t

        embed_obs_batch = torch.cat([embed_obs.unsqueeze(0), self.predicted_embed_obses])[
            : self.max_imagination_steps
        ]  # (batch, dim)
        hidden_batch = torch.cat(
            [self.exact_forward_dynamics_hidden_state.unsqueeze(0), self.forward_dynamics_hidden_states]
        )[
            : self.max_imagination_steps
        ]  # (batch, depth, dim)

        action_dist_batch: Distribution = self.policy_net(embed_obs_batch, hidden_batch)
        value_dist_batch: Distribution = self.value_net(embed_obs_batch, hidden_batch)
        action_batch, value_batch = action_dist_batch.sample(), value_dist_batch.sample()
        action_log_prob_batch = action_dist_batch.log_prob(action_batch)

        pred_obs_dist_batch, _, _, next_hidden_batch = self.forward_dynamics(
            embed_obs_batch, hidden_batch, action_batch
        )
        pred_obs_batch = pred_obs_dist_batch.sample()

        self.step_data[DataKeys.ACTION] = action_batch[0]  # a_t
        self.step_data[DataKeys.ACTION_LOG_PROBABILITY] = action_log_prob_batch[0]  # log \pi(a_t | o_t, h_t)
        self.step_data[DataKeys.VALUE] = value_batch[0]  # v_t
        self.step_data[DataKeys.HIDDEN] = self.exact_forward_dynamics_hidden_state  # h_t
        self.logger.log("agent/value", value_batch[0])

        self.predicted_embed_obs_dists = pred_obs_dist_batch
        self.predicted_embed_obses = pred_obs_batch
        self.forward_dynamics_hidden_states = next_hidden_batch
        self.exact_forward_dynamics_hidden_state = next_hidden_batch[0]

        self.logger.update()

        return action_batch[0]

    def setup(self, observation: Tensor) -> Tensor:
        super().setup(observation)
        self.step_data = StepData()
        self.forward_dynamics_hidden_states = torch.empty(0).type_as(self.exact_forward_dynamics_hidden_state)
        self.predicted_embed_obses = torch.empty(0).type_as(self.exact_forward_dynamics_hidden_state)

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
