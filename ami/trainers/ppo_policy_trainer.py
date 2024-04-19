from functools import partial
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.utils.data import DataLoader
from typing_extensions import override

from ami.tensorboard_loggers import StepIntervalLogger

from ..data.buffers.buffer_names import BufferNames
from ..data.buffers.ppo_trajectory_buffer import PPOTrajectoryBuffer
from ..data.interfaces import ThreadSafeDataUser
from ..models.model_names import ModelNames
from ..models.model_wrapper import ModelWrapper
from ..models.policy_value_common_net import PolicyValueCommonNet
from .base_trainer import BaseTrainer


class StepOutput(TypedDict):
    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    approx_kl: Tensor
    clipfrac: Tensor


class PPOPolicyTrainer(BaseTrainer):
    """Trainer for ppo policy.

    Reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
    From: https://github.com/MLShukai/PrimitiveAMI/blob/main/src/models/ppo_lit_module.py
    """

    def __init__(
        self,
        partial_dataloader: partial[DataLoader[Tensor]],
        partial_optimizer: partial[torch.optim.Optimizer],
        device: torch.device,
        logger: StepIntervalLogger,
        max_epochs: int = 1,
        minimum_dataset_size: int = 1,
        norm_advantage: bool = True,
        clip_coef: float = 0.1,
        clip_vloss: bool = True,
        entropy_coef: float = 0.001,
        vfunc_coef: float = 0.5,
    ) -> None:
        """Initializes a PPOLitModule.

        Args:
            partial_dataloader: A partially instantiated DataLoader lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) used for training.
            max_epochs: The number of times to use all of the dataset.
            minimum_dataset_size: Minimum dataset size required to consider the trainer in a trainable state.
            norm_advantage: Toggles normalization of advantages.
            clip_coef: The coefficient for surrogate clipping.
            clip_vloss: Toggles the use of a clipped loss for the value function, as per the paper.
            entropy_coef: The coefficient for entropy.
            vfunc_coef: The coefficient for the value function.
        """
        super().__init__()

        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.device = device
        self.logger = logger
        self.logger_state = self.logger.state_dict()
        self.max_epochs = max_epochs
        self.minimum_dataset_size = minimum_dataset_size
        self.norm_advantage = norm_advantage
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.entropy_coef = entropy_coef
        self.vfunc_coef = vfunc_coef

    def on_data_users_dict_attached(self) -> None:
        super().on_data_users_dict_attached()
        self.trajectory_data_user: ThreadSafeDataUser[PPOTrajectoryBuffer] = self.get_data_user(
            BufferNames.PPO_TRAJECTORY
        )

    def on_model_wrappers_dict_attached(self) -> None:
        super().on_model_wrappers_dict_attached()
        self.policy_value: ModelWrapper[PolicyValueCommonNet] = self.get_training_model(ModelNames.POLICY_VALUE)

        self.optimizer_state = self.partial_optimizer(self.policy_value.parameters()).state_dict()

    def is_trainable(self) -> bool:
        self.trajectory_data_user.update()
        return self.trajectory_data_user.buffer.dataset_size >= self.minimum_dataset_size

    def model_forward(self, obs: Tensor) -> tuple[Distribution, Tensor]:
        """Written for type annotation."""
        return self.policy_value(obs)

    def training_step(self, batch: tuple[Tensor, ...]) -> StepOutput:
        """Perform a single training step on a batch of data."""
        obses, actions, logprobs, advantanges, returns, values = batch

        new_action_dist, new_values = self.model_forward(obses)
        new_logprobs = new_action_dist.log_prob(actions)
        entropy = new_action_dist.entropy()

        logratio = new_logprobs - logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - logratio).mean()
            clipfracs = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

        if self.norm_advantage:
            advantanges = (advantanges - advantanges.mean()) / (advantanges.std() + 1e-8)

        if advantanges.ndim == 1:
            advantanges = advantanges.unsqueeze(1)

        # Policy loss
        pg_loss1 = -advantanges * ratio
        pg_loss2 = -advantanges * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        new_values = new_values.flatten()
        if self.clip_vloss:
            v_loss_unclipped = (new_values - returns) ** 2
            v_clipped = values + torch.clamp(new_values - values, -self.clip_coef, self.clip_coef)
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((new_values - returns) ** 2).mean()

        entropy_loss = entropy.mean()

        loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.vfunc_coef

        # Output
        output = StepOutput(
            loss=loss,
            policy_loss=pg_loss,
            value_loss=v_loss,
            entropy=entropy,
            approx_kl=approx_kl,
            clipfrac=clipfracs,
        )
        return output

    def train(self) -> None:
        self.policy_value.to(self.device)

        optimizer = self.partial_optimizer(self.policy_value.parameters())
        optimizer.load_state_dict(self.optimizer_state)
        self.logger.load_state_dict(self.logger_state)
        dataset = self.trajectory_data_user.get_dataset()
        dataloader = self.partial_dataloader(dataset=dataset)

        for _ in range(self.max_epochs):
            for batch in dataloader:
                batch = [d.to(self.device) for d in batch]
                out = self.training_step(batch)
                self.logger.log("ppo_policy/loss", out["loss"])
                self.logger.log("ppo_policy/policy_loss", out["policy_loss"])
                self.logger.log("ppo_policy/value_loss", out["value_loss"])
                # self.logger.log("ppo_policy/entropy", out["entropy"])
                self.logger.log("ppo_policy/approx_kl", out["approx_kl"])
                self.logger.log("ppo_policy/clipfrac", out["clipfrac"])

                optimizer.zero_grad()
                out["loss"].backward()
                optimizer.step()
                self.logger.update()

        self.optimizer_state = optimizer.state_dict()

    def teardown(self) -> None:
        super().teardown()
        self.trajectory_data_user.clear()  # Can not use old buffer data because ppo is on-policy method.

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.optimizer_state, path / "optimizer.pt")
        torch.save(self.logger_state, path / "logger.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.optimizer_state = torch.load(path / "optimizer.pt")
        self.logger_state = torch.load(path / "logger.pt")
