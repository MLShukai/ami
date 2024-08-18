from functools import partial
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.interfaces import ThreadSafeDataUser
from ami.logger import get_training_thread_logger
from ami.models.components.mixture_desity_network import NormalMixture
from ami.models.forward_dynamics import ForwardDynamcisWithActionReward
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.policy_or_value_network import PolicyOrValueNetwork
from ami.tensorboard_loggers import StepIntervalLogger

from .base_trainer import BaseTrainer


def lambda_no_modify_lr(epoch: int) -> float:
    return 1.0


class ImaginationTrajectory(TypedDict):
    """TypedDict which contains the outputs of imagination."""

    observations: Tensor  # o_0:H+1
    hiddens: Tensor  # h_0:H+1
    actions: Tensor  # a_0:H
    action_entropies: Tensor  # Ha_0:H
    rewards: Tensor  # r_1:H+1
    next_values: Tensor  # v_1:H+1


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
        partial_policy_lr_scheduler: partial[LRScheduler] | None = None,
        partial_value_lr_scheduler: partial[LRScheduler] | None = None,
        max_epochs: int = 1,
        imagination_trajectory_length: int = 1,
        discount_factor: float = 0.99,  # gamma for rl
        eligibility_trace_decay: float = 0.95,  # for lambda-return
        entropy_coef: float = 0.001,
        imagination_temperature: float = 1.0,
        minimum_dataset_size: int = 1,
        minimum_new_data_count: int = 0,
        update_start_train_count: int = 0,
    ) -> None:
        """Initializes an Dreaming PolicyValueTrainer object.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_policy_optimizer: A partially instantiated optimizer for policy lacking provided parameters.
            partial_value_optimizer: A partially instantiated optimizer for value lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            logger: A StepIntervalLogger object for logging training progress.
            partial_policy_lr_scheduler: A partially instantiated lr scheduler for the policy network optimizer.
            partial_value_lr_scheduler: A partially instantiated lr scheduler for the value network optimizer.
            max_epochs: Maximum number of epochs for training.
            imagination_trajectory_length: The length of dreaming steps.
            discount_factor: Discount factor (gamma) for reinforcement learning.
            eligibility_trace_decay: Decay factor (lambda) for eligibility trace in lambda-return calculation.
            entropy_coef: Coefficient for entropy regularization in policy loss.
            imagination_temperature: The sampling uncertainty for forward dynamics prediction (mixture density network only.)
            minimum_dataset_size: Minimum size of the dataset required to start training.
            minimum_new_data_count: Minimum number of new data count required to run the training.
            update_start_train_count: Actual traning procedure will be start from the training count larger then this value.
        """
        super().__init__()
        self.partial_data_loader = partial_dataloader
        self.partial_policy_optimizer = partial_policy_optimizer
        self.partial_value_optimizer = partial_value_optimizer
        self.device = device
        self.logger = logger
        self.max_epochs = max_epochs
        self.partial_policy_lr_scheduler = (
            partial_policy_lr_scheduler
            if partial_policy_lr_scheduler is not None
            else partial(LambdaLR, lr_lambda=lambda_no_modify_lr)
        )
        self.partial_value_lr_scheduler = (
            partial_value_lr_scheduler
            if partial_value_lr_scheduler is not None
            else partial(LambdaLR, lr_lambda=lambda_no_modify_lr)
        )
        self.imagination_trajectory_length = imagination_trajectory_length
        self.discount_factor = discount_factor
        self.eligibility_trace_decay = eligibility_trace_decay
        self.entropy_coef = entropy_coef
        self.imagination_temperature = imagination_temperature
        self.minimum_dataset_size = minimum_dataset_size
        self.minimum_new_data_count = minimum_new_data_count
        self.update_start_train_count = update_start_train_count
        self._current_train_count = 0

        self.console_logger = get_training_thread_logger(self.__class__.__name__)

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

        policy_optim = self.partial_policy_optimizer(self.policy_net.parameters())
        value_optim = self.partial_value_optimizer(self.value_net.parameters())
        self.policy_optimizer_state = policy_optim.state_dict()
        self.value_optimizer_state = value_optim.state_dict()

        self.policy_lr_scheduler_state = self.partial_policy_lr_scheduler(optimizer=policy_optim).state_dict()  # type: ignore
        self.value_lr_scheduler_state = self.partial_value_lr_scheduler(value_optim).state_dict()  # type: ignore

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

        # Setup schedulers
        policy_lr_scheduler = self.partial_policy_lr_scheduler(policy_optimizer)  # type: ignore
        policy_lr_scheduler.load_state_dict(self.policy_lr_scheduler_state)
        value_lr_scheduler = self.partial_value_lr_scheduler(value_optimizer)  # type: ignore
        value_lr_scheduler.load_state_dict(self.value_lr_scheduler_state)

        # Setup dataset
        dataset = self.initial_states_data_user.get_dataset()
        dataloader = self.partial_data_loader(dataset=dataset)

        # Training.
        self._current_train_count += 1
        if self._current_train_count <= self.update_start_train_count:
            self.console_logger.info(
                f"Training is skipped because training count {self._current_train_count-1} (start: {self.update_start_train_count})"
            )
            return

        prefix = "dreaming_policy_value/"
        for _ in range(self.max_epochs):
            for batch in dataloader:
                # Initial states setup.
                observation, hidden = batch
                observation = observation.to(self.device)
                hidden = hidden.to(self.device)

                # Value network is used only inference in the imagination.
                self.value_net.freeze_model()

                trajectory = self.imagine_trajectory(initial_state=(observation, hidden))

                returns = compute_lambda_return(
                    trajectory["rewards"], trajectory["next_values"], self.discount_factor, self.eligibility_trace_decay
                )

                # Update policy network
                policy_optimizer.zero_grad()
                entropy_loss = trajectory["action_entropies"].mean()
                return_loss = returns.mean()
                policy_loss = -(return_loss + entropy_loss * self.entropy_coef)  # maximize.
                policy_loss.backward()
                policy_grad_norm = torch.cat(
                    [p.grad.flatten() for p in self.policy_net.parameters() if p.grad is not None]
                ).norm()
                policy_optimizer.step()

                # Stop gradient for learning value network.
                observations = trajectory["observations"].detach()
                hiddens = trajectory["hiddens"].detach()
                returns = returns.detach()

                # Update value network.
                self.value_net.unfreeze_model()
                value_optimizer.zero_grad()
                value_losses = []
                for i in range(self.imagination_trajectory_length):
                    value_dist: Distribution = self.value_net(observations[i], hiddens[i])
                    value_losses.append(-value_dist.log_prob(returns[i]).mean())
                value_loss = torch.mean(torch.stack(value_losses))
                value_loss.backward()
                value_grad_norm = torch.cat(
                    [p.grad.flatten() for p in self.value_net.parameters() if p.grad is not None]
                ).norm()
                value_optimizer.step()

                # Logging
                self.logger.log(prefix + "return", return_loss)
                self.logger.log(prefix + "entropy", entropy_loss)
                self.logger.log(prefix + "policy_loss", policy_loss)
                self.logger.log(prefix + "value_loss", value_loss)
                self.logger.log(prefix + "policy_grad_norm", policy_grad_norm)
                self.logger.log(prefix + "value_grad_norm", value_grad_norm)
                self.logger.update()

            # Updating LR Schedulers
            policy_lr_scheduler.step()
            value_lr_scheduler.step()
            self.logger.log(prefix + "policy_lr", policy_lr_scheduler.get_last_lr()[0])
            self.logger.log(prefix + "value_lr", value_lr_scheduler.get_last_lr()[0])

        self.policy_optimizer_state = policy_optimizer.state_dict()
        self.value_optimizer_state = value_optimizer.state_dict()
        self.policy_lr_scheduler_state = policy_lr_scheduler.state_dict()
        self.value_lr_scheduler_state = value_lr_scheduler.state_dict()

    def imagine_trajectory(self, initial_state: tuple[Tensor, Tensor]) -> ImaginationTrajectory:
        """Imagines a trajectory of states, actions, and rewards using the
        current policy and value networks.

        This method uses the policy network to generate actions, the forward dynamics model to predict
        next states and rewards, and the value network to estimate future values. It simulates a
        trajectory of length `imagination_trajectory_length` starting from the given initial state.

        Args:
            initial_state: A tuple containing the initial observation and hidden state tensors.

        Returns:
            ImaginationTrajectory: A dict containing the following tensors, each with the first dimension
            equal to the imagination trajectory length or its +1:
                - observations: Predicted observations for each step.
                - hiddens: Hidden states for each step.
                - actions: Actions taken at each step.
                - action_entropies: Entropies of the action distributions at each step.
                - rewards: Predicted rewards for each step.
                - next_values: Estimated values of the next states.
        """
        observation, hidden = initial_state

        # Setup buffers
        observations = [observation]  # o_0:H+1
        hiddens = [hidden]  # h_0:H+1
        actions = []  # a_0:H
        action_entropies = []
        rewards = []  # r_1:H+1
        next_values = []  # v_1:H+1

        for _ in range(self.imagination_trajectory_length):  # H step

            # Take one step.
            action_dist: Distribution = self.policy_net(observation, hidden)
            action = action_dist.rsample()

            next_obs_dist: Distribution
            reward_dist: Distribution
            next_hidden: Tensor
            next_obs_dist, _, reward_dist, next_hidden = self.forward_dynamics(observation, hidden, action)

            if isinstance(next_obs_dist, NormalMixture):
                observation = next_obs_dist.rsample(temperature=self.imagination_temperature)
            else:
                observation = next_obs_dist.rsample()

            if isinstance(reward_dist, NormalMixture):
                reward = reward_dist.rsample(temperature=self.imagination_temperature)
            else:
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

        return ImaginationTrajectory(
            observations=torch.stack(observations),
            hiddens=torch.stack(hiddens),
            actions=torch.stack(actions),
            action_entropies=torch.stack(action_entropies),
            rewards=torch.stack(rewards),
            next_values=torch.stack(next_values),
        )

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.policy_optimizer_state, path / "policy_optimizer.pt")
        torch.save(self.value_optimizer_state, path / "value_optimizer.pt")
        torch.save(self.policy_lr_scheduler_state, path / "policy_lr_scheduler.pt")
        torch.save(self.value_lr_scheduler_state, path / "value_lr_scheduler.pt")
        torch.save(self.logger.state_dict(), path / "logger.pt")
        torch.save(self._current_train_count, path / "current_train_count.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.policy_optimizer_state = torch.load(path / "policy_optimizer.pt")
        self.value_optimizer_state = torch.load(path / "value_optimizer.pt")
        self.policy_lr_scheduler_state = torch.load(path / "policy_lr_scheduler.pt")
        self.value_lr_scheduler_state = torch.load(path / "value_lr_scheduler.pt")
        self.logger.load_state_dict(torch.load(path / "logger.pt"))
        self._current_train_count = torch.load(path / "current_train_count.pt")


def compute_lambda_return(
    rewards: Tensor, next_values: Tensor, discount_factor: float, eligibility_trace_decay: float
) -> Tensor:
    """Computes the lambda return for a given sequence of rewards and estimated
    next state values.

    The lambda return is a mixture of n-step returns for different n, weighted by the eligibility
    trace decay factor (lambda). It provides a balance between bias and variance in the return estimation.

    Args:
        rewards: Tensor of shape (L, *) containing the rewards for each step. L is the sequence length.
        next_values: Tensor of shape (L, *) containing the estimated values of the next states.
        discount_factor: The discount factor (gamma) for future rewards (0 < gamma <= 1).
        eligibility_trace_decay: The decay factor (lambda) for the eligibility trace (0 <= lambda <= 1).

    Returns:
        lambda_returns: Tensor of shape (L, *) containing the computed lambda returns for each step.
            The first dimension corresponds to the time steps, starting from t=0 to t=L-1.

    Note:
        The shapes of rewards and next_values should be identical.
        The * in the shape represents any number of additional dimensions (e.g., batch size).
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


class InitialMultiplicationLRScheduler(LambdaLR):
    """A custom learning rate scheduler that applies a multiplication factor to
    the learning rate for a specified number of initial epochs.

    This scheduler allows you to modify the learning rate by a given factor for a set number of epochs
    at the beginning of training, after which it reverts to the original learning rate. This can be
    useful for various training strategies, such as:

    1. Warm-up: Use a factor > 1 to gradually increase the learning rate.
    2. Initial suppression: Use a factor < 1 to start with a lower learning rate.
    3. Freezing: Use a factor of 0 to temporarily freeze learning for some layers.

    Example:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # To suppress initial learning rate:
        scheduler = InitialMultiplicationLRScheduler(optimizer, multiplication_factor=0.1, initial_epochs=5)
        # To implement a warm-up:
        # scheduler = InitialMultiplicationLRScheduler(optimizer, multiplication_factor=10, initial_epochs=5)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        multiplication_factor: float,
        initial_epochs: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): The optimizer whose learning rate should be scheduled.
            multiplication_factor (float): The factor by which to multiply the learning rate during the initial phase.
            initial_epochs (int): The number of epochs to apply the modified learning rate.
            last_epoch (int, optional): The index of the last epoch. Default: -1.
            verbose (bool, optional): If True, prints a message to stdout for each update. Default: False.
        """

        def lr_lambda(epoch: int) -> float:
            return multiplication_factor if epoch < initial_epochs else 1.0

        super().__init__(optimizer, lr_lambda, last_epoch, verbose)
