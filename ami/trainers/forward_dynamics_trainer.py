import time
from functools import partial
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.causal_data_buffer import CausalDataBuffer
from ami.data.interfaces import ThreadSafeDataUser
from ami.models.forward_dynamics import (
    ForwardDynamcisWithActionReward,
    ForwardDynamics,
    PrimitiveForwardDynamics,
)
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.tensorboard_loggers import StepIntervalLogger

from .base_trainer import BaseTrainer
from .components.random_time_series_sampler import RandomTimeSeriesSampler


class ForwardDynamicsTrainer(BaseTrainer):
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[torch.Tensor]],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
        logger: StepIntervalLogger,
        observation_encoder_name: ModelNames | None = None,
        max_epochs: int = 1,
        minimum_dataset_size: int = 2,
        minimum_new_data_count: int = 0,
    ) -> None:
        """Initializes an ForwardDynamicsTrainer object.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            minimum_new_data_count: Minimum number of new data count required to run the training.
        """
        super().__init__()
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.device = device
        self.logger = logger
        self.observation_encoder_name = observation_encoder_name
        self.max_epochs = max_epochs
        assert minimum_dataset_size >= 2, "minimum_dataset_size must be at least 2"
        self.minimum_dataset_size = minimum_dataset_size
        self.minimum_new_data_count = minimum_new_data_count
        self.dataset_previous_get_time = float("-inf")

    def on_data_users_dict_attached(self) -> None:
        self.trajectory_data_user: ThreadSafeDataUser[CausalDataBuffer] = self.get_data_user(
            BufferNames.FORWARD_DYNAMICS_TRAJECTORY
        )

    def on_model_wrappers_dict_attached(self) -> None:
        self.forward_dynamics: ModelWrapper[ForwardDynamics] = self.get_training_model(ModelNames.FORWARD_DYNAMICS)
        self.optimizer_state = self.partial_optimizer(self.forward_dynamics.parameters()).state_dict()
        if self.observation_encoder_name is None:
            self.observation_encoder = None
        else:
            self.observation_encoder = self.get_frozen_model(self.observation_encoder_name)

    def is_trainable(self) -> bool:
        self.trajectory_data_user.update()
        return len(self.trajectory_data_user.buffer) >= self.minimum_dataset_size and self._is_new_data_available()

    def _is_new_data_available(self) -> bool:
        return (
            self.trajectory_data_user.buffer.count_data_added_since(self.dataset_previous_get_time)
            >= self.minimum_new_data_count
        )

    def get_dataset(self) -> Dataset[Tensor]:
        dataset = self.trajectory_data_user.get_dataset()
        self.dataset_previous_get_time = time.time()
        return dataset

    def train(self) -> None:
        self.forward_dynamics.to(self.device)

        optimizer = self.partial_optimizer(self.forward_dynamics.parameters())
        optimizer.load_state_dict(self.optimizer_state)

        dataloader = self.partial_dataloader(dataset=self.get_dataset())

        for _ in range(self.max_epochs):
            for batch in dataloader:
                observations, hiddens, actions = batch
                if self.observation_encoder is not None:
                    with torch.no_grad():
                        observations = self.observation_encoder.infer(observations)

                observations = observations.to(self.device)

                observations, hidden, actions, observations_next = (
                    observations[:-1],
                    hiddens[0],
                    actions[:-1],
                    observations[1:],
                )

                hidden = hidden.to(self.device)
                actions = actions.to(self.device)

                optimizer.zero_grad()
                observations_next_hat_dist, _ = self.forward_dynamics(observations, hidden, actions)
                loss = -observations_next_hat_dist.log_prob(observations_next).mean()
                self.logger.log("forward_dynamics/loss", loss)
                loss.backward()
                optimizer.step()
                self.logger.update()

        self.optimizer_state = optimizer.state_dict()

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.optimizer_state, path / "optimizer.pt")
        torch.save(self.logger.state_dict(), path / "logger.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.optimizer_state = torch.load(path / "optimizer.pt")
        self.logger.load_state_dict(torch.load(path / "logger.pt"))


class ForwardDynamicsWithActionRewardTrainer(BaseTrainer):
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[torch.Tensor]],
        partial_sampler: partial[RandomTimeSeriesSampler],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
        logger: StepIntervalLogger,
        observation_encoder_name: ModelNames | None = None,
        max_epochs: int = 1,
        minimum_dataset_size: int = 2,
        minimum_new_data_count: int = 0,
        obs_loss_coef: float = 1.0,
        action_loss_coef: float = 1.0,
        reward_loss_coef: float = 1.0,
        gradient_clip_norm: float | None = None,
    ) -> None:
        """Initialization.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_sampler: A partially instantiated sampler lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            minimum_new_data_count: Minimum number of new data count required to run the training.
        """
        super().__init__()
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.partial_sampler = partial_sampler
        self.device = device
        self.logger = logger
        self.observation_encoder_name = observation_encoder_name
        self.max_epochs = max_epochs
        assert minimum_dataset_size >= 2, "minimum_dataset_size must be at least 2"
        self.minimum_dataset_size = minimum_dataset_size
        self.minimum_new_data_count = minimum_new_data_count
        self.obs_loss_coef = obs_loss_coef
        self.action_loss_coef = action_loss_coef
        self.reward_loss_coef = reward_loss_coef
        self.gradient_clip_norm = gradient_clip_norm
        self.dataset_previous_get_time = float("-inf")

    def on_data_users_dict_attached(self) -> None:
        self.trajectory_data_user: ThreadSafeDataUser[CausalDataBuffer] = self.get_data_user(
            BufferNames.FORWARD_DYNAMICS_TRAJECTORY
        )

    def on_model_wrappers_dict_attached(self) -> None:
        self.forward_dynamics: ModelWrapper[ForwardDynamcisWithActionReward] = self.get_training_model(
            ModelNames.FORWARD_DYNAMICS
        )
        self.optimizer_state = self.partial_optimizer(self.forward_dynamics.parameters()).state_dict()
        if self.observation_encoder_name is None:
            self.observation_encoder = None
        else:
            self.observation_encoder = self.get_frozen_model(self.observation_encoder_name)

    def is_trainable(self) -> bool:
        self.trajectory_data_user.update()
        return len(self.trajectory_data_user.buffer) >= self.minimum_dataset_size and self._is_new_data_available()

    def _is_new_data_available(self) -> bool:
        return (
            self.trajectory_data_user.buffer.count_data_added_since(self.dataset_previous_get_time)
            >= self.minimum_new_data_count
        )

    def get_dataset(self) -> Dataset[Tensor]:
        dataset = self.trajectory_data_user.get_dataset()
        self.dataset_previous_get_time = time.time()
        return dataset

    def train(self) -> None:
        self.forward_dynamics.to(self.device)
        if self.observation_encoder is not None:
            self.observation_encoder.to(self.device)

        optimizer = self.partial_optimizer(self.forward_dynamics.parameters())
        optimizer.load_state_dict(self.optimizer_state)

        dataset = self.get_dataset()
        sampler = self.partial_sampler(dataset)
        dataloader = self.partial_dataloader(dataset=dataset, sampler=sampler)

        for _ in range(self.max_epochs):
            for batch in dataloader:
                observations, hiddens, actions, rewards = batch
                if self.observation_encoder is not None:
                    with torch.no_grad():
                        batch_time_shape = observations.shape[:2]
                        observations = self.observation_encoder.infer(observations.flatten(0, 1))
                        output_shape = batch_time_shape + observations.shape[1:]
                        observations = observations.reshape(output_shape)
                observations = observations.to(self.device)
                actions = actions.to(self.device)

                observations, hidden, actions, observations_next, actions_next, rewards = (
                    observations[:, :-1],  # o_0:T-1
                    hiddens[:, 0],  # h_0
                    actions[:, :-1],  # a_0:T-1
                    observations[:, 1:],  # o_1:T
                    actions[:, 1:],  # a_1:T
                    rewards[:, :-1],  # r_1:T because rewards are always t+1.
                )

                hidden = hidden.to(self.device)
                rewards = rewards.to(self.device)

                optimizer.zero_grad()

                observations_next_hat_dist: Distribution
                actions_next_hat_dist: Distribution
                reward_hat_dist: Distribution
                observations_next_hat_dist, actions_next_hat_dist, reward_hat_dist, _ = self.forward_dynamics(
                    observations, hidden, actions
                )

                observation_loss = -observations_next_hat_dist.log_prob(observations_next).mean()
                action_loss = -actions_next_hat_dist.log_prob(actions_next).mean()
                reward_loss = -reward_hat_dist.log_prob(rewards).mean()

                loss = (
                    self.obs_loss_coef * observation_loss
                    + self.action_loss_coef * action_loss
                    + self.reward_loss_coef * reward_loss
                )
                prefix = "forward_dynamics/"
                self.logger.log(prefix + "loss", loss)
                self.logger.log(prefix + "observation_loss", observation_loss)
                self.logger.log(prefix + "action_loss", action_loss)
                self.logger.log(prefix + "reward_loss", reward_loss)

                loss.backward()

                grad_norm = grad_norm = torch.cat(
                    [p.grad.flatten() for p in self.forward_dynamics.parameters() if p.grad is not None]
                ).norm()
                self.logger.log(prefix + "metrics/grad_norm", grad_norm)

                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.forward_dynamics.parameters(), self.gradient_clip_norm, error_if_nonfinite=True
                    )
                optimizer.step()
                self.logger.update()

        self.optimizer_state = optimizer.state_dict()

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.optimizer_state, path / "optimizer.pt")
        torch.save(self.logger.state_dict(), path / "logger.pt")
        torch.save(self.dataset_previous_get_time, path / "dataset_previous_get_time.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.optimizer_state = torch.load(path / "optimizer.pt")
        self.logger.load_state_dict(torch.load(path / "logger.pt"))
        self.dataset_previous_get_time = torch.load(path / "dataset_previous_get_time.pt")


class PrimitiveForwardDynamicsTrainer(BaseTrainer):
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[torch.Tensor]],
        partial_sampler: partial[RandomTimeSeriesSampler],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
        logger: StepIntervalLogger,
        max_epochs: int = 1,
        minimum_new_data_count: int = 2,
    ) -> None:
        """Initialization.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_sampler: A partially instantiated sampler lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            minimum_new_data_count: Minimum number of new data count required to run the training.
        """
        super().__init__()
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.partial_sampler = partial_sampler
        self.device = device
        self.logger = logger
        self.max_epochs = max_epochs
        assert minimum_new_data_count >= 2, "minimum_new_data_count must be at least 2"
        self.minimum_new_data_count = minimum_new_data_count
        self.dataset_previous_get_time = float("-inf")

    def on_data_users_dict_attached(self) -> None:
        self.trajectory_data_user: ThreadSafeDataUser[CausalDataBuffer] = self.get_data_user(
            BufferNames.FORWARD_DYNAMICS_TRAJECTORY
        )

    def on_model_wrappers_dict_attached(self) -> None:
        self.forward_dynamics: ModelWrapper[PrimitiveForwardDynamics] = self.get_training_model(
            ModelNames.FORWARD_DYNAMICS
        )
        self.optimizer_state = self.partial_optimizer(self.forward_dynamics.parameters()).state_dict()

    def is_trainable(self) -> bool:
        self.trajectory_data_user.update()
        return self._is_new_data_available()

    def _is_new_data_available(self) -> bool:
        return (
            self.trajectory_data_user.buffer.count_data_added_since(self.dataset_previous_get_time)
            >= self.minimum_new_data_count
        )

    def get_dataset(self) -> Dataset[Tensor]:
        dataset = self.trajectory_data_user.get_dataset()
        self.dataset_previous_get_time = time.time()
        return dataset

    def train(self) -> None:
        self.forward_dynamics.to(self.device)

        optimizer = self.partial_optimizer(self.forward_dynamics.parameters())
        optimizer.load_state_dict(self.optimizer_state)

        dataset = self.get_dataset()
        sampler = self.partial_sampler(dataset)
        dataloader = self.partial_dataloader(dataset=dataset, sampler=sampler)

        for _ in range(self.max_epochs):
            for batch in dataloader:
                observations, actions = batch
                observations = observations.to(self.device)
                actions = actions.to(self.device)

                observations, actions, observations_next = (
                    observations[:, :-1],  # o_0:T-1
                    actions[:, :-1],  # a_0:T-1
                    observations[:, 1:],  # o_1:T
                )

                optimizer.zero_grad()

                observations_next_hat_dist: Distribution
                observations_next_hat_dist = self.forward_dynamics(observations, actions)

                loss = -observations_next_hat_dist.log_prob(observations_next).mean()

                prefix = "forward_dynamics/"
                self.logger.log(prefix + "loss", loss)

                loss.backward()

                grad_norm = grad_norm = torch.cat(
                    [p.grad.flatten() for p in self.forward_dynamics.parameters() if p.grad is not None]
                ).norm()
                self.logger.log(prefix + "metrics/grad_norm", grad_norm)
                optimizer.step()
                self.logger.update()

        self.optimizer_state = optimizer.state_dict()

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.optimizer_state, path / "optimizer.pt")
        torch.save(self.logger.state_dict(), path / "logger.pt")
        torch.save(self.dataset_previous_get_time, path / "dataset_previous_get_time.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.optimizer_state = torch.load(path / "optimizer.pt")
        self.logger.load_state_dict(torch.load(path / "logger.pt"))
        self.dataset_previous_get_time = torch.load(path / "dataset_previous_get_time.pt")


class ImaginingForwardDynamicsTrainer(BaseTrainer):
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[torch.Tensor]],
        partial_sampler: partial[RandomTimeSeriesSampler],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
        logger: StepIntervalLogger,
        max_epochs: int = 1,
        sequence_length: int = 1,
        imagination_length: int = 1,
        minimum_new_data_count: int = 0,
        imagination_average_method: Callable[[Tensor], Tensor] = torch.mean,
    ) -> None:
        """Initialization.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_sampler: A partially instantiated sampler lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            sequence_length: The length of context sequence (not used as imagination)
            imagination_length: The length of imagination horizon.
            minimum_new_data_count: Minimum number of new data count required to run the training.
        """
        super().__init__()
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.partial_sampler = partial_sampler
        self.device = device
        self.logger = logger
        self.max_epochs = max_epochs
        if imagination_length < 1:
            raise ValueError("imagination_length must be larger than 0")
        if sequence_length < 1:
            raise ValueError("sequence_length must be larger than 0")
        self.imagination_length = imagination_length
        self.sequence_length = sequence_length
        self._sample_sequence_length = imagination_length + sequence_length
        self.minimum_new_data_count = minimum_new_data_count
        self.imagination_average_method = imagination_average_method

        self.dataset_previous_get_time = float("-inf")

    def on_data_users_dict_attached(self) -> None:
        self.trajectory_data_user: ThreadSafeDataUser[CausalDataBuffer] = self.get_data_user(
            BufferNames.FORWARD_DYNAMICS_TRAJECTORY
        )

    def on_model_wrappers_dict_attached(self) -> None:
        self.forward_dynamics: ModelWrapper[ForwardDynamics] = self.get_training_model(ModelNames.FORWARD_DYNAMICS)
        self.optimizer_state = self.partial_optimizer(self.forward_dynamics.parameters()).state_dict()

    def is_trainable(self) -> bool:
        self.trajectory_data_user.update()
        return len(self.trajectory_data_user.buffer) >= self._sample_sequence_length and self._is_new_data_available()

    def _is_new_data_available(self) -> bool:
        return (
            self.trajectory_data_user.buffer.count_data_added_since(self.dataset_previous_get_time)
            >= self.minimum_new_data_count
        )

    def get_dataset(self) -> Dataset[Tensor]:
        dataset = self.trajectory_data_user.get_dataset()
        self.dataset_previous_get_time = time.time()
        return dataset

    def train(self) -> None:
        self.forward_dynamics.to(self.device)

        optimizer = self.partial_optimizer(self.forward_dynamics.parameters())
        optimizer.load_state_dict(self.optimizer_state)

        dataset = self.get_dataset()
        sampler = self.partial_sampler(dataset, sequence_length=self._sample_sequence_length)  # to sample target
        dataloader = self.partial_dataloader(dataset=dataset, sampler=sampler)

        for _ in range(self.max_epochs):
            batch: tuple[Tensor, ...]
            for batch in dataloader:
                observations, hiddens, actions = batch
                if observations.shape[:2] != actions.shape[:2]:
                    raise ValueError("The shape between observations and actions is not consistent!")
                if observations.size(1) != self._sample_sequence_length:
                    raise ValueError("Time length must be imagination_length + sequence_length!")

                observations = observations.to(self.device)  # (B, T+H, *)
                actions = actions.to(self.device)  # (B, T+H)
                obs_imaginations, hiddens = (
                    observations[:, : self.sequence_length],  # o_0:T, (B, T, *)
                    hiddens[:, 0].to(self.device),  # h_-1, (B, *)
                )
                optimizer.zero_grad()

                obses_next_hat_dist: Distribution
                loss_imaginations: list[Tensor] = []
                for i in range(self.imagination_length):
                    action_imaginations = actions[:, i : self.sequence_length + i]  # a_i:i+T, (B, T, *)
                    obs_targets = observations[:, i + 1 : self.sequence_length + i + 1]  # o_i+1:T+i+1, (B, T, *)

                    if i > 0:
                        action_imaginations = action_imaginations.flatten(0, 1)  # (B', *)
                        obs_targets = obs_targets.flatten(0, 1)  # (B', *)

                    obses_next_hat_dist, next_hiddens = self.forward_dynamics(
                        obs_imaginations, hiddens, action_imaginations
                    )
                    loss = -obses_next_hat_dist.log_prob(obs_targets).mean()
                    loss_imaginations.append(loss)
                    obs_imaginations = obses_next_hat_dist.rsample()

                    if i == 0:
                        obs_imaginations = obs_imaginations.flatten(0, 1)  # (B, T, *) -> (B', *)
                        hiddens = next_hiddens.movedim(2, 1).flatten(
                            0, 1
                        )  # h'_i, (B, D, T, *) -> (B, T, D, *) -> (B', D, *)

                loss = self.imagination_average_method(torch.stack(loss_imaginations))
                loss.backward()

                prefix = "forward_dynamics/"
                self.logger.log(prefix + "loss", loss)
                for i, loss_item in enumerate(loss_imaginations, start=1):
                    self.logger.log(prefix + f"loss/imagination_{i}", loss_item)

                grad_norm = grad_norm = torch.cat(
                    [p.grad.flatten() for p in self.forward_dynamics.parameters() if p.grad is not None]
                ).norm()
                self.logger.log(prefix + "metrics/grad_norm", grad_norm)
                optimizer.step()
                self.logger.update()

        self.optimizer_state = optimizer.state_dict()

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.optimizer_state, path / "optimizer.pt")
        torch.save(self.logger.state_dict(), path / "logger.pt")
        torch.save(self.dataset_previous_get_time, path / "dataset_previous_get_time.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.optimizer_state = torch.load(path / "optimizer.pt")
        self.logger.load_state_dict(torch.load(path / "logger.pt"))
        self.dataset_previous_get_time = torch.load(path / "dataset_previous_get_time.pt")
