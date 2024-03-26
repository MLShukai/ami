from functools import partial

import torch
from torch.nn.functional import mse_loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.causal_data_buffer import CausalDataBuffer
from ami.data.interfaces import ThreadSafeDataUser
from ami.models.components.forward_dynamics import ForwardDynamics
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper

from .base_trainer import BaseTrainer


class ForwardDynamicsTrainer(BaseTrainer):
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[torch.Tensor]],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
    ):
        super().__init__()
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.device = device
        self.batch_size: int = partial_dataloader.keywords["batch_size"]

    def on_data_users_dict_attached(self) -> None:
        self.trajectory_data_user: ThreadSafeDataUser[CausalDataBuffer] = self.get_data_user(
            BufferNames.FORWARD_DYNAMICS_TRAJECTORY
        )

    def on_model_wrappers_dict_attached(self) -> None:
        self.forward_dynamics: ModelWrapper[ForwardDynamics] = self.get_training_model(ModelNames.FORWARD_DYNAMICS)
        self.optimizer_state = self.partial_optimizer(self.forward_dynamics.parameters()).state_dict()

    def is_trainable(self) -> bool:
        self.trajectory_data_user.update()
        return len(self.trajectory_data_user.buffer) >= self.batch_size

    def train(self) -> None:
        optimizer = self.partial_optimizer(self.forward_dynamics.parameters())
        optimizer.load_state_dict(self.optimizer_state)
        dataset = self.trajectory_data_user.get_dataset()
        dataloader = self.partial_dataloader(dataset=dataset)

        for batch in dataloader:
            embed_observations, embed_observation_nexts, actions, hiddens = batch
            embed_observations = embed_observations.to(self.device)
            embed_observation_nexts = embed_observation_nexts.to(self.device)
            actions = actions.to(self.device)
            hidden = hiddens[0].to(self.device)
            optimizer.zero_grad()

            embed_observation_nexts_hat_dist, _ = self.forward_dynamics(embed_observations, actions, hidden)
            embed_observation_nexts_hat = embed_observation_nexts_hat_dist.rsample()
            loss = mse_loss(embed_observation_nexts_hat, embed_observation_nexts)
            loss.backward()
            optimizer.step()

        self.optimizer_state = optimizer.state_dict()
