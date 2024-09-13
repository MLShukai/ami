from collections import deque
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2.functional
from torch import Tensor
from torch.distributions import Distribution
from typing_extensions import override

from ami.tensorboard_loggers import TimeIntervalLogger

from ...data.step_data import DataKeys, StepData
from ...models.forward_dynamics import ForwardDynamcisWithActionReward
from ...models.model_names import ModelNames
from ...models.model_wrapper import ThreadSafeInferenceWrapper
from ...models.policy_or_value_network import PolicyOrValueNetwork
from ...models.policy_value_common_net import PolicyValueCommonNet
from .base_agent import BaseAgent
from .utils import PolicyValueCommonProxy


class MultiStepImaginationCuriosityImageAgent(BaseAgent[Tensor, Tensor]):
    def __init__(
        self,
        initial_hidden: Tensor,
        logger: TimeIntervalLogger,
        reward_average_method: Callable[[Tensor], Tensor],
        max_imagination_steps: int = 1,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
        log_reward_imaginations: bool = True,
        log_reward_imaginations_every_n_steps: int = 1,
        log_reward_imaginations_max_history_size: int = 1,
        log_reward_imaginations_append_interval: int = 1,
        # 再構成画像の可視化ログについて
        log_reconstruction_imaginations: bool = True,
        log_reconstruction_imaginations_every_n_steps: int = 1,
        log_reconstruction_imaginations_max_history_size: int = 1,
        log_reconstruction_imaginations_append_interval: int = 1,
    ) -> None:
        """Constructs Agent.

        Args:
            initial_hidden: Initial hidden state for the forward dynamics model.
            reward_average_method: The method for averaging rewards that predicted through multi imaginations.
                Input is reward (imagination, ), and return value must be scalar.
            max_imagination_steps: Max step for imagination.
            log_reward_imaginations: Flag to enable logging of reward imaginations.
            log_reward_imaginations_every_n_steps: Number of steps between each logging of reward imaginations.
            log_reward_imaginations_max_history_size: Maximum number of reward imagination entries to keep in the log history.
            log_reward_imaginations_append_interval: Number of steps between each append to the reward imaginations log.
            log_reconstruction_imaginations: Flag to enable logging of reconstruction imaginations.
            log_reconstruction_imaginations_every_n_steps: Number of steps between each logging of reconstruction imaginations.
            log_reconstruction_imaginations_max_history_size: Maximum number of reconstruction imagination entries to keep in the log history.
            log_reconstruction_imaginations_append_interval: Number of steps between each append to the reconstruction imaginations log.
        """
        super().__init__()
        assert max_imagination_steps > 0

        self.exact_forward_dynamics_hidden_state = initial_hidden
        self.logger = logger
        self.reward_average_method = reward_average_method
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift
        self.max_imagination_steps = max_imagination_steps

        self.log_reward_imaginations = log_reward_imaginations
        self.log_reward_imaginations_every_n_steps = log_reward_imaginations_every_n_steps
        self.reward_imaginations_deque: deque[npt.NDArray[Any]] = deque(maxlen=log_reward_imaginations_max_history_size)
        self.reward_imaginations_global_step_deque: deque[int] = deque(maxlen=log_reward_imaginations_max_history_size)
        self.log_reward_imaginations_append_interval = log_reward_imaginations_append_interval

        # 観測の再構成画像ログについて
        self.log_reconstruction_imaginations = log_reconstruction_imaginations
        self.log_reconstruction_imaginations_every_n_steps = log_reconstruction_imaginations_every_n_steps
        self.log_reconstruction_imaginations_append_interval = log_reconstruction_imaginations_append_interval
        self.reconstruction_imaginations_deque: deque[Tensor] = deque(
            maxlen=log_reconstruction_imaginations_max_history_size
        )
        self.reconstruction_imaginations_ground_truth_deque: deque[Tensor] = deque(
            maxlen=log_reconstruction_imaginations_max_history_size
        )

    @property
    def global_step(self) -> int:
        return self.logger.global_step

    def on_inference_models_attached(self) -> None:
        super().on_inference_models_attached()
        self.image_encoder: ThreadSafeInferenceWrapper[nn.Module] = self.get_inference_model(ModelNames.IMAGE_ENCODER)
        self.forward_dynamics: ThreadSafeInferenceWrapper[ForwardDynamcisWithActionReward] = self.get_inference_model(
            ModelNames.FORWARD_DYNAMICS
        )

        self.policy_value_net: ThreadSafeInferenceWrapper[PolicyValueCommonNet] | PolicyValueCommonProxy
        if self.check_model_exists(ModelNames.POLICY_VALUE):
            self.policy_value_net = self.get_inference_model(ModelNames.POLICY_VALUE)
        else:
            policy_net: ThreadSafeInferenceWrapper[PolicyOrValueNetwork] = self.get_inference_model(ModelNames.POLICY)
            value_net: ThreadSafeInferenceWrapper[PolicyOrValueNetwork] = self.get_inference_model(ModelNames.VALUE)
            self.policy_value_net = PolicyValueCommonProxy(policy_net, value_net)

        self.image_decoder: ThreadSafeInferenceWrapper[nn.Module] | None = None
        if self.check_model_exists(ModelNames.IMAGE_DECODER):
            self.image_decoder = self.get_inference_model(ModelNames.IMAGE_DECODER)

    # ------ Interaction Process ------
    exact_forward_dynamics_hidden_state: Tensor  # (depth, dim)
    predicted_embed_obs_dist_imaginations: Distribution  # (imaginations, dim)
    predicted_embed_obs_imaginations: Tensor  # (imaginations, dim)
    forward_dynamics_hidden_state_imaginations: Tensor  # (imaginations, depth, dim)
    step_data: StepData

    def _common_step(self, observation: Tensor, initial_step: bool = False) -> Tensor:
        """Common step procedure for agent.

        If `initial_step` is False, some procedures are skipped.
        """
        embed_obs: Tensor = self.image_encoder(observation)

        if not initial_step:
            # 報酬計算は初期ステップではできないためスキップ。
            embed_obs = embed_obs.type(self.predicted_embed_obs_imaginations.dtype)
            embed_obs = embed_obs.to(self.predicted_embed_obs_imaginations.device)
            target_obses = embed_obs.expand_as(self.predicted_embed_obs_imaginations)
            reward_imaginations = (
                -self.predicted_embed_obs_dist_imaginations.log_prob(target_obses).flatten(1).mean(-1)
                * self.reward_scale
                + self.reward_shift
            )
            reward = self.reward_average_method(reward_imaginations)
            self.logger.log("agent/reward", reward)
            # for i, r in enumerate(reward_imaginations, start=1):
            #     self.logger.log(f"agent/reward_{i}step", r)

            # ステップの冒頭でデータコレクトすることで前ステップのデータを収集する。
            self.step_data[DataKeys.REWARD] = reward
            self.data_collectors.collect(self.step_data)

            if self.log_reward_imaginations:
                self.reward_imaginations_logging_step(reward_imaginations)

            if self.log_reconstruction_imaginations and self.image_decoder is not None:
                self.reconstruction_imaginations_logging_step(observation, self.image_decoder)

        self.step_data[DataKeys.OBSERVATION] = observation  # o_t
        self.step_data[DataKeys.EMBED_OBSERVATION] = embed_obs  # z_t

        embed_obs_imaginations = torch.cat([embed_obs.unsqueeze(0), self.predicted_embed_obs_imaginations])[
            : self.max_imagination_steps
        ]  # (imaginations, dim)
        hidden_imaginations = torch.cat(
            [self.exact_forward_dynamics_hidden_state.unsqueeze(0), self.forward_dynamics_hidden_state_imaginations]
        )[
            : self.max_imagination_steps
        ]  # (imaginations, depth, dim)

        action_dist_imaginations: Distribution
        value_imaginations: Tensor
        action_dist_imaginations, value_imaginations = self.policy_value_net(
            embed_obs_imaginations, hidden_imaginations
        )
        action_imaginations = action_dist_imaginations.sample()
        action_log_prob_imaginations = action_dist_imaginations.log_prob(action_imaginations)

        pred_obs_dist_imaginations, _, _, next_hidden_imaginations = self.forward_dynamics(
            embed_obs_imaginations, hidden_imaginations, action_imaginations
        )
        pred_obs_imaginations = pred_obs_dist_imaginations.sample()

        self.step_data[DataKeys.ACTION] = action_imaginations[0]  # a_t
        self.step_data[DataKeys.ACTION_LOG_PROBABILITY] = action_log_prob_imaginations[0]  # log \pi(a_t | o_t, h_t)
        self.step_data[DataKeys.VALUE] = value_imaginations[0]  # v_t
        self.step_data[DataKeys.HIDDEN] = self.exact_forward_dynamics_hidden_state  # h_t
        self.logger.log("agent/value", value_imaginations[0])

        self.predicted_embed_obs_dist_imaginations = pred_obs_dist_imaginations
        self.predicted_embed_obs_imaginations = pred_obs_imaginations
        self.forward_dynamics_hidden_state_imaginations = next_hidden_imaginations
        self.exact_forward_dynamics_hidden_state = next_hidden_imaginations[0]

        self.logger.update()

        return action_imaginations[0]

    def setup(self, observation: Tensor) -> Tensor:
        super().setup(observation)
        self.step_data = StepData()

        device = self.exact_forward_dynamics_hidden_state.device
        dtype = self.exact_forward_dynamics_hidden_state.dtype
        self.forward_dynamics_hidden_state_imaginations = torch.empty(0, device=device, dtype=dtype)
        self.predicted_embed_obs_imaginations = torch.empty(0, device=device)

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
            path / "exact_forward_dynamics_hidden_state.pt",
            map_location=self.exact_forward_dynamics_hidden_state.device,
        )

    def reward_imaginations_logging_step(self, reward_imaginations: Tensor) -> None:
        """Logs reward imaginations data and visualizes it at specified
        intervals.

        Args:
            reward_imaginations: Tensor containing reward imagination values.
        """
        # 長期的予測の誤差値とそのステップの格納
        if (
            reward_imaginations.size(0) == self.max_imagination_steps
            and self.global_step % self.log_reward_imaginations_append_interval == 0
        ):
            self.reward_imaginations_deque.append(reward_imaginations.cpu().numpy())
            self.reward_imaginations_global_step_deque.append(self.global_step)

        # 長期的予測の誤差の可視化
        if (
            self.global_step % self.log_reward_imaginations_every_n_steps == 0
            and len(self.reward_imaginations_deque) > 0
        ):
            self.visualize_reward_imaginations()

    BASE_FIG_SIZE = 0.6
    ADJUST_FIG_WIDTH = 5
    COLOR_MAP = "plasma"

    def visualize_reward_imaginations(self) -> None:
        """Creates and logs a heatmap visualization of reward imaginations.

        This method generates a heatmap using the collected reward imagination data,
        where each row represents a different global step and each column represents
        an imagination step. The heatmap is then logged to TensorBoard for visual analysis.

        The heatmap provides insights into how the predicted rewards change over time
        and across different imagination steps, helping to track the agent's
        performance and the accuracy of its reward predictions.
        """

        imaginations_history_size = len(self.reward_imaginations_deque)
        figsize = (
            self.BASE_FIG_SIZE * self.max_imagination_steps + self.ADJUST_FIG_WIDTH,
            self.BASE_FIG_SIZE * imaginations_history_size,
        )

        data = np.stack(self.reward_imaginations_deque)[::-1]
        xticklabels = np.arange(self.max_imagination_steps) + 1
        yticklabels = list(reversed(self.reward_imaginations_global_step_deque))

        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
        seaborn.heatmap(
            data=data,
            ax=ax,
            annot=True,
            cmap=self.COLOR_MAP,
            linewidths=0.5,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )
        ax.set_xlabel("imagination steps")
        ax.set_ylabel("global steps")

        self.logger.tensorboard.add_figure("agent/multistep-imagination-errors", fig, self.global_step)

    def reconstruction_imaginations_logging_step(
        self, observation: Tensor, image_decoder: ThreadSafeInferenceWrapper[nn.Module]
    ) -> None:
        """Logs reconstruction imaginations and visualizes them at specified
        intervals.

        Args:
            observation: Current observation tensor.
            image_decoder: Image decoder model for reconstructions.
        """
        # 長期的予測の再構成画像とそのGround Truthの格納
        if (
            self.predicted_embed_obs_imaginations.size(0) == self.max_imagination_steps
            and self.global_step % self.log_reconstruction_imaginations_append_interval == 0
        ):
            self.reconstruction_imaginations_ground_truth_deque.append(observation.cpu())
            reconstructions: Tensor = image_decoder(self.predicted_embed_obs_imaginations)
            self.reconstruction_imaginations_deque.append(reconstructions.cpu())

        # 長期予測の再構成画像の可視化
        if (
            self.global_step % self.log_reconstruction_imaginations_every_n_steps == 0
            and len(self.reconstruction_imaginations_deque) > 0
        ):
            self.visualize_reconstruction_imaginations()

    def visualize_reconstruction_imaginations(self) -> None:
        """Visualizes the reconstruction imaginations.

        This method creates a grid of images showing the ground truth observations
        alongside the reconstructed observations for each imagination step. The grid
        is then logged to TensorBoard for visual analysis.

        The visualization helps in understanding how well the agent's imagination
        process is reconstructing observations over multiple steps, providing insights
        into the quality of the agent's internal world model.
        """

        reconstructions = torch.stack(list(self.reconstruction_imaginations_deque))  # (H, T, C, H, W)
        image_size = reconstructions.size()[-2:]
        ground_truth = torch.stack(list(self.reconstruction_imaginations_ground_truth_deque))  # (H, C, H, W)
        ground_truth = torchvision.transforms.v2.functional.resize(ground_truth, image_size)

        log_images = torch.cat([ground_truth.unsqueeze(1), reconstructions], dim=1)  # (H, T+1, C, H, W)
        log_images = log_images.flatten(0, 1)  # # ((H * T+1), C, H, W)
        grid_image = torchvision.utils.make_grid(log_images, self.max_imagination_steps + 1, normalize=True)

        self.logger.tensorboard.add_image("agent/multistep-imagination-recontructions", grid_image, self.global_step)


def average_exponentially(rewards: Tensor, decay: float) -> Tensor:
    """Averages rewards by exponential decay style.

    Args:
        rewards: shape (imaginations, )
        decay: The exponential decay factor.

    Returns:
        Tensor: averaged reward tensor.
    """
    assert 0 <= decay < 1
    assert rewards.ndim == 1

    decay_factors = decay ** torch.arange(len(rewards), device=rewards.device, dtype=rewards.dtype)

    equi_series_sum = (1 - decay ** len(rewards)) / (1 - decay)  # the sum of an equi-series

    return torch.sum(rewards * decay_factors, dim=0) / equi_series_sum
