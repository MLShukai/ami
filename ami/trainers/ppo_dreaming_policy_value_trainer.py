import time
from functools import partial
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.interfaces import ThreadSafeDataUser
from ami.logger import get_training_thread_logger
from ami.models.components.mixture_desity_network import NormalMixture
from ami.models.forward_dynamics import ForwardDynamcisWithActionReward
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.policy_value_common_net import PolicyValueCommonNet
from ami.tensorboard_loggers import StepIntervalLogger

from .base_trainer import BaseTrainer


class ImaginationTrajectory(TypedDict):
    """TypedDict which contains the outputs of imagination."""

    observations: Tensor  # o_0:H
    hiddens: Tensor  # h_0:H
    actions: Tensor  # a_0:H
    action_log_probs: Tensor  # log pi 0:H
    rewards: Tensor  # r_1:H+1
    values: Tensor  # v_0:H
    advantages: Tensor  # A_0:H
    returns: Tensor  # g_0:H


class PPODreamingPolicyValueTrainer(BaseTrainer):
    """Training the policy and value model by Dreamer and PPO method.

    This class combines elements from Dreamer (for world model learning and imagination)
    and PPO (for policy optimization) to train a policy and value network. It uses the
    world model to generate imagined trajectories, which are then used to update the
    policy using PPO's objective.

    References:
        Dreamer V1
        "Dream to Control: Learning Behaviors by Latent Imagination",
        arXiv:1912.01603, 2019.

        Dreamer V3
        "Mastering Diverse Domains through World Models",
        arXiv:2301.04104, 2023.

        PPO
        "Proximal Policy Optimization Algorithms",
        arXiv:1707.06347, 2017.
    """

    @override
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[Tensor]],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
        logger: StepIntervalLogger,
        max_epochs: int = 1,
        imagination_trajectory_length: int = 1,
        imagination_temperature: float = 1.0,
        discount_factor: float = 0.99,  # gamma for rl
        eligibility_trace_decay: float = 0.95,  # for lambda-return
        normalize_advantage: bool = True,
        clip_coef: float = 0.1,
        clip_value_loss: bool = True,
        entropy_coef: float = 0.001,
        value_func_coef: float = 0.5,
        minimum_dataset_size: int = 1,
        minimum_new_data_count: int = 0,
        update_start_train_count: int = 0,
    ) -> None:
        """Initializes an Dreaming PolicyValueTrainer object.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer for policy and value networks lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            logger: A StepIntervalLogger object for logging training progress.
            max_epochs: Maximum number of epochs for training.
            imagination_trajectory_length: The length of dreaming steps.
            discount_factor: Discount factor (gamma) for reinforcement learning.
            eligibility_trace_decay: Decay factor (lambda) for eligibility trace in generalized advantage esitimation.
            normalize_advantage: Whether to normalize the advantage.
            clip_coef: Clipping parameter for PPO.
            clip_value_loss: Whether to clip the value loss.
            entropy_coef: Coefficient for entropy regularization in policy loss.
            value_func_coef: Coefficient for value function loss.
            imagination_temperature: The sampling uncertainty for forward dynamics prediction (mixture density network only.)
            minimum_dataset_size: Minimum size of the dataset required to start training.
            minimum_new_data_count: Minimum number of new data count required to run the training.
            update_start_train_count: Actual traning procedure will be start from the training count larger then this value.
        """
        super().__init__()
        self.partial_data_loader = partial_dataloader
        self.partial_optimizer = partial_optimizer
        self.device = device
        self.logger = logger
        self.max_epochs = max_epochs
        self.imagination_trajectory_length = imagination_trajectory_length
        self.imagination_temperature = imagination_temperature
        self.discount_factor = discount_factor
        self.eligibility_trace_decay = eligibility_trace_decay
        self.normalize_advantage = normalize_advantage
        self.clip_coef = clip_coef
        self.clip_value_loss = clip_value_loss
        self.entropy_coef = entropy_coef
        self.value_func_coef = value_func_coef
        self.minimum_dataset_size = minimum_dataset_size
        self.minimum_new_data_count = minimum_new_data_count
        self.update_start_train_count = update_start_train_count
        self._current_train_count = 0

        self.console_logger = get_training_thread_logger(self.__class__.__name__)
        self.dataset_previous_get_time = float("-inf")

    @override
    def on_data_users_dict_attached(self) -> None:
        self.initial_states_data_user: ThreadSafeDataUser[RandomDataBuffer] = self.get_data_user(
            BufferNames.DREAMING_INITIAL_STATES
        )

    @override
    def on_model_wrappers_dict_attached(self) -> None:
        self.forward_dynamics: ModelWrapper[ForwardDynamcisWithActionReward] = self.get_frozen_model(
            ModelNames.FORWARD_DYNAMICS
        )

        self.policy_value_net: ModelWrapper[PolicyValueCommonNet] = self.get_training_model(ModelNames.POLICY_VALUE)
        self.optimizer_state = self.partial_optimizer(self.policy_value_net.parameters()).state_dict()

    @override
    def is_trainable(self) -> bool:
        self.initial_states_data_user.update()
        return self._is_minimum_data_available() and self._is_new_data_available()

    def _is_minimum_data_available(self) -> bool:
        return len(self.initial_states_data_user.buffer) >= self.minimum_dataset_size

    def _is_new_data_available(self) -> bool:
        return (
            self.initial_states_data_user.buffer.count_data_added_since(self.dataset_previous_get_time)
            >= self.minimum_new_data_count
        )

    def get_dataset(self) -> Dataset[Tensor]:
        dataset = self.initial_states_data_user.get_dataset()
        self.dataset_previous_get_time = time.time()
        return dataset

    @override
    def train(self) -> None:
        # Setup model device.
        self.forward_dynamics.to(self.device)
        self.policy_value_net.to(self.device)

        # Setup optimizers.
        optimizer = self.partial_optimizer(self.policy_value_net.parameters())
        optimizer.load_state_dict(self.optimizer_state)

        # Setup dataset
        dataloader = self.partial_data_loader(dataset=self.get_dataset())

        # Training.
        self._current_train_count += 1
        if self._current_train_count <= self.update_start_train_count:
            self.console_logger.info(
                f"Training is skipped because training count {self._current_train_count-1} (start: {self.update_start_train_count})"
            )
            return

        prefix = "ppo_dreaming_policy_value/"

        for _ in range(self.max_epochs):
            for batch in dataloader:
                # Initial states setup.
                observation, hidden = batch
                observation = observation.to(self.device)
                hidden = hidden.to(self.device)

                # Imagine trajectory.
                trajectory = self.imagine_trajectory(initial_state=(observation, hidden))

                # Compute losses.
                output = self.ppo_step(trajectory)

                # Update
                optimizer.zero_grad()
                output["loss"].backward()
                optimizer.step()

                # Logging
                for name, value in output.items():
                    self.logger.log(prefix + name, value)
                self.logger.log(prefix + "mean_return", trajectory["returns"].mean())

                self.logger.update()

        self.optimizer_state = optimizer.state_dict()

    @torch.no_grad()
    def imagine_trajectory(self, initial_state: tuple[Tensor, Tensor]) -> ImaginationTrajectory:
        """Imagines a trajectory of states, actions, and rewards using the
        current policy and value networks.

        This method uses the policy network to generate actions, the forward dynamics model to predict
        next states and rewards, and the value network to estimate future values. It simulates a
        trajectory of length `imagination_trajectory_length` starting from the given initial state.

        NOTE: In PPO, Computing policy gradient with reinforce, so do not use forward dynamics gradient.

        Args:
            initial_state: A tuple containing the initial observation and hidden state tensors.

        Returns:
            ImaginationTrajectory: A dict containing the following tensors, each with the first dimension
            equal to the `imagination_trajectory_length * batch_size`.
        """
        observation, hidden = initial_state

        # Setup buffers
        observations = []  # o_0:H
        hiddens = []  # h_0:H
        actions = []  # a_0:H
        action_log_probs = []
        rewards = []  # r_1:H+1
        values = []  # v_0:H

        for _ in range(self.imagination_trajectory_length):  # H step

            # Take one step.
            action_dist: Distribution
            value: Tensor

            action_dist, value = self.policy_value_net(observation, hidden)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)

            # Add to list
            actions.append(action)
            action_log_probs.append(action_log_prob)
            observations.append(observation)
            hiddens.append(hidden)
            values.append(value)

            next_obs_dist: Distribution
            reward_dist: Distribution
            next_hidden: Tensor
            next_obs_dist, _, reward_dist, next_hidden = self.forward_dynamics(observation, hidden, action)

            if isinstance(next_obs_dist, NormalMixture):
                observation = next_obs_dist.sample(temperature=self.imagination_temperature)
            else:
                observation = next_obs_dist.sample()

            if isinstance(reward_dist, NormalMixture):
                reward = reward_dist.sample(temperature=self.imagination_temperature)
            else:
                reward = reward_dist.sample()
            hidden = next_hidden

            # Add to list
            rewards.append(reward)

        rewards_tensor = torch.stack(rewards)
        values_tensor = torch.stack(values)
        advantages = compute_advantage(
            rewards_tensor,
            values_tensor,
            self.policy_value_net(observation, hidden)[1],
            self.discount_factor,
            self.eligibility_trace_decay,
        )
        returns = advantages + values_tensor

        return ImaginationTrajectory(
            observations=torch.stack(observations).flatten(end_dim=1),
            hiddens=torch.stack(hiddens).flatten(end_dim=1),
            actions=torch.stack(actions).flatten(end_dim=1),
            action_log_probs=torch.stack(action_log_probs).flatten(end_dim=1),
            rewards=rewards_tensor.flatten(end_dim=1),
            values=values_tensor.flatten(end_dim=1),
            advantages=advantages.flatten(end_dim=1),
            returns=returns.flatten(end_dim=1),
        )

    def ppo_step(self, trajectory: ImaginationTrajectory) -> dict[str, Tensor]:
        """Perform a single ppo training step on trajectory data."""
        obses, hiddens, actions, logprobs, advantanges, returns, values = (
            trajectory["observations"],
            trajectory["hiddens"],
            trajectory["actions"],
            trajectory["action_log_probs"],
            trajectory["advantages"],
            trajectory["returns"],
            trajectory["values"],
        )

        new_action_dist: Distribution
        new_values: Tensor
        new_action_dist, new_values = self.policy_value_net(obses, hiddens)
        new_logprobs = new_action_dist.log_prob(actions)
        entropy = new_action_dist.entropy()

        logratio = new_logprobs - logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - logratio).mean()
            clipfracs = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

        if self.normalize_advantage:
            advantanges = (advantanges - advantanges.mean()) / (advantanges.std() + 1e-8)

        # expanding advantages for broadcasting.
        for _ in range(ratio.ndim - advantanges.ndim):
            advantanges = advantanges.unsqueeze(-1)

        # Policy loss
        pg_loss1 = -advantanges * ratio
        pg_loss2 = -advantanges * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        if self.clip_value_loss:
            v_loss_unclipped = (new_values - returns) ** 2
            v_clipped = values + torch.clamp(new_values - values, -self.clip_coef, self.clip_coef)
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((new_values - returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.value_func_coef

        # Output
        output = {
            "loss": loss,
            "policy_loss": pg_loss,
            "value_loss": v_loss,
            "entropy": entropy_loss,
            "approx_kl": approx_kl,
            "clipfrac": clipfracs,
        }

        return output

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.optimizer_state, path / "optimizer.pt")
        torch.save(self.logger.state_dict(), path / "logger.pt")
        torch.save(self._current_train_count, path / "current_train_count.pt")
        torch.save(self.dataset_previous_get_time, path / "dataset_previous_get_time.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.optimizer_state = torch.load(path / "optimizer.pt")
        self.logger.load_state_dict(torch.load(path / "logger.pt"))
        self._current_train_count = torch.load(path / "current_train_count.pt")
        self.dataset_previous_get_time = torch.load(path / "dataset_previous_get_time.pt")


def compute_advantage(
    rewards: Tensor, values: Tensor, final_next_value: Tensor, gamma: float, gae_lambda: float
) -> Tensor:
    """Compute advantages from values.

    Args:
        rewards: shape (step length, *)
        values: shape (step length, *)
        final_next_value: shape (*, )
        gamma: Discount factor.
        gae_lambda: The lambda of generalized advantage estimation.

    Returns:
        advantages: shape (step length, *)
    """
    advantages = torch.empty_like(values)

    lastgaelam = torch.tensor(0.0, device=values.device, dtype=values.dtype)

    for t in reversed(range(values.size(0))):
        if t == values.size(0) - 1:
            nextvalues = final_next_value
        else:
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * lastgaelam

    return advantages
