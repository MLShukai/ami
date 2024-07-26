from functools import partial
from pathlib import Path

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.interfaces import ThreadSafeDataUser
from ami.models.forward_dynamics import ForwardDynamcisWithActionReward
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.policy_or_value_network import PolicyOrValueNetwork
from ami.tensorboard_loggers import StepIntervalLogger

from .base_trainer import BaseTrainer


class DreamingPolicyValueTrainer(BaseTrainer):
    """Training the policy and value model by Dreamer method.

    References:
        Dreamer V1
        "Dream to Control: Learning Behaviors by Latent Imagination",
        arXiv:1912.01603, 2019.
    """

    @override
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[Tensor]],
        partial_policy_optimizer: partial[Optimizer],
        partial_value_optimizer: partial[Optimizer],
        device: torch.device,
        logger: StepIntervalLogger,
        max_epochs: int = 1,
        imagination_trajectory_length: int = 1,
        discount_factor: float = 0.99,  # gamma for rl
        eligibility_trace_decay: float = 0.95,  # for lambda-return
        entropy_coef: float = 0.001,
        minimum_dataset_size: int = 1,
        minimum_new_data_count: int = 0,
    ) -> None:
        """Initializes an Dreaming PolicyValueTrainer object.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_policy_optimizer: A partially instantiated optimizer for policy lacking provided parameters.
            partial_value_optimizer: A partially instantiated optimizer for value lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            imagination_trajectory_length: The length of dreaming steps.
            ...
            minimum_new_data_count: Minimum number of new data count required to run the training.
        """
        super().__init__()
        self.partial_data_loader = partial_dataloader
        self.partial_policy_optimizer = partial_policy_optimizer
        self.partial_value_optimizer = partial_value_optimizer
        self.device = device
        self.logger = logger
        self.max_epochs = max_epochs
        self.imagination_trajectory_length = imagination_trajectory_length
        self.discount_factor = discount_factor
        self.eligibility_trace_decay = eligibility_trace_decay
        self.entropy_coef = entropy_coef
        self.minimum_dataset_size = minimum_dataset_size
        self.minimum_new_data_count = minimum_new_data_count

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

        self.policy_net: ModelWrapper[PolicyOrValueNetwork] = self.get_training_model(ModelNames.POLICY)
        self.value_net: ModelWrapper[PolicyOrValueNetwork] = self.get_training_model(ModelNames.VALUE)

        self.policy_optimizer_state = self.partial_policy_optimizer(self.policy_net.parameters()).state_dict()
        self.value_optimizer_state = self.partial_value_optimizer(self.value_net.parameters()).state_dict()

    @override
    def is_trainable(self) -> bool:
        self.initial_states_data_user.update()
        return self._is_minimum_data_available() and self._is_new_data_available()

    def _is_minimum_data_available(self) -> bool:
        return len(self.initial_states_data_user.buffer) >= self.minimum_dataset_size

    def _is_new_data_available(self) -> bool:
        return self.initial_states_data_user.buffer.new_data_count >= self.minimum_new_data_count

    @override
    def train(self) -> None:
        # Setup model device.
        self.forward_dynamics.to(self.device)
        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        # Setup optimizers.
        policy_optimizer = self.partial_policy_optimizer(self.policy_net.parameters())
        policy_optimizer.load_state_dict(self.policy_optimizer_state)
        value_optimizer = self.partial_value_optimizer(self.value_net.parameters())
        value_optimizer.load_state_dict(self.value_optimizer_state)

        # Setup dataset
        dataset = self.initial_states_data_user.get_dataset()
        dataloader = self.partial_data_loader(dataset=dataset)

        # Training.
        for _ in range(self.max_epochs):
            for batch in dataloader:
                # Initial states setup.
                observation, hidden = batch
                observation = observation.to(self.device)
                hidden = hidden.to(self.device)

                observations, hiddens, _, action_entropies, rewards, next_values = self.imagine_trajectory(
                    initial_state=(observation, hidden)
                )

                returns = compute_lambda_return(
                    rewards, next_values, self.discount_factor, self.eligibility_trace_decay
                )

                # Update policy network
                policy_optimizer.zero_grad()
                entropy_loss = action_entropies.mean()
                return_loss = returns.sum(0).mean()
                policy_loss = -(return_loss + entropy_loss * self.entropy_coef)  # maximize.
                policy_loss.backward()
                policy_optimizer.step()

                # Update value network.
                value_optimizer.zero_grad()
                value_losses: list[Tensor] = []
                for i in range(self.imagination_trajectory_length):
                    obs, hidden = observations[i].detach(), hiddens[i].detach()
                    target = returns[i].detach()
                    value_dist: Distribution = self.value_net(obs, hidden)
                    value_losses.append(-value_dist.log_prob(target).mean())
                value_loss = torch.mean(torch.stack(value_losses))
                value_loss.backward()
                value_optimizer.step()

                # Logging
                prefix = "dreaming_policy_value/"
                self.logger.log(prefix + "return", return_loss)
                self.logger.log(prefix + "entropy", entropy_loss)
                self.logger.log(prefix + "policy_loss", policy_loss)
                self.logger.log(prefix + "value_loss", value_loss)
                self.logger.update()

        self.policy_optimizer_state = policy_optimizer.state_dict()
        self.value_optimizer_state = value_optimizer.state_dict()

    def imagine_trajectory(self, initial_state: tuple[Tensor, Tensor]) -> tuple[Tensor, ...]:
        """
        Args:
            initial_state: observation and hidden state.

        Returns:
            tuple[Tensor, ...]: observations, hiddens, actions, action_entopies, rewards, next_values.
                The first dimension is imagination trajectory length.

        """
        observation, hidden = initial_state

        # Setup buffers
        observations: list[Tensor] = [observation]  # o_0:H+1
        hiddens: list[Tensor] = [hidden]  # h_0:H+1
        actions: list[Tensor] = []  # a_0:H
        action_entropies: list[Tensor] = []
        rewards: list[Tensor] = []  # r_1:H+1
        next_values: list[Tensor] = []  # v_1:H+1

        for _ in range(self.imagination_trajectory_length):  # H step

            # Take one step.
            action_dist: Distribution = self.policy_net(observation, hidden)
            action = action_dist.rsample()

            next_obs_dist: Distribution
            reward_dist: Distribution
            next_hidden: Tensor
            next_obs_dist, _, reward_dist, next_hidden = self.forward_dynamics(observation, hidden, action)

            observation = next_obs_dist.rsample()
            reward = reward_dist.rsample()
            hidden = next_hidden

            value_dist: Distribution = self.value_net(observation, hidden)
            next_value = value_dist.rsample()

            # Add to list
            actions.append(action)
            action_entropies.append(action_dist.entropy())
            observations.append(observation)
            hiddens.append(hidden)
            rewards.append(reward)
            next_values.append(next_value)

        return (
            torch.stack(observations),
            torch.stack(hiddens),
            torch.stack(actions),
            torch.stack(action_entropies),
            torch.stack(rewards),
            torch.stack(next_values),
        )

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.policy_optimizer_state, path / "policy_optimizer.pt")
        torch.save(self.value_optimizer_state, path / "value_optimizer.pt")
        torch.save(self.logger.state_dict(), path / "logger.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.policy_optimizer_state = torch.load(path / "policy_optimizer.pt")
        self.value_optimizer_state = torch.load(path / "value_optimizer.pt")
        self.logger.load_state_dict(torch.load(path / "logger.pt"))


def compute_lambda_return(
    rewards: Tensor, next_values: Tensor, discount_factor: float, eligibility_trace_decay: float
) -> Tensor:
    """Computes the lambda Return.

    Args:
        rewards: r_1:T+1, (L, *)
        next_values: v_1:T+1, (L, *)
        discount_factor: gamma.
        eligibility_trace_decay: lambda.

    Returns:
        lambda_returns: g_0:T, (L, *).
    """
    assert rewards.shape == next_values.shape

    lambda_returns = []
    last_lambda_return = next_values[-1]

    for i in reversed(range(rewards.size(0))):
        last_lambda_return = rewards[i] + discount_factor * (
            (1 - eligibility_trace_decay) * next_values[i] + eligibility_trace_decay * last_lambda_return
        )
        lambda_returns.append(last_lambda_return)

    return torch.stack(list(reversed(lambda_returns)))
