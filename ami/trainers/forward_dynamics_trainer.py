from functools import partial
from pathlib import Path

import torch
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.causal_data_buffer import CausalDataBuffer
from ami.data.interfaces import ThreadSafeDataUser
from ami.models.forward_dynamics import ForwardDynamics
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.tensorboard_loggers import StepIntervalLogger

from .base_trainer import BaseTrainer


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
    ) -> None:
        """Initializes an ForwardDynamicsTrainer object.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
        """
        super().__init__()
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.device = device
        self.logger = logger
        self.logger_state = self.logger.state_dict()
        self.observation_encoder_name = observation_encoder_name
        self.max_epochs = max_epochs
        assert minimum_dataset_size >= 2, "minimum_dataset_size must be at least 2"
        self.minimum_dataset_size = minimum_dataset_size

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
        return len(self.trajectory_data_user.buffer) >= self.minimum_dataset_size

    def train(self) -> None:
        self.forward_dynamics.to(self.device)

        optimizer = self.partial_optimizer(self.forward_dynamics.parameters())
        optimizer.load_state_dict(self.optimizer_state)
        self.logger.load_state_dict(self.logger_state)
        dataset = self.trajectory_data_user.get_dataset()
        dataloader = self.partial_dataloader(dataset=dataset)

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
        torch.save(self.logger_state, path / "logger.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.optimizer_state = torch.load(path / "optimizer.pt")
        self.logger_state = torch.load(path / "logger.pt")
