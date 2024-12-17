import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Distribution
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.multimodal_temporal_data_buffer import (
    MultimodalTemporalDataBuffer,
)
from ami.data.interfaces import ThreadSafeDataUser
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.temporal_encoder import MultimodalTemporalEncoder
from ami.tensorboard_loggers import StepIntervalLogger

from .base_trainer import BaseTrainer


class MultimodalTemporalEncoderTrainer(BaseTrainer):
    """Trainer for MultimodalTemporalEncoder that handles multiple input
    modalities."""

    def __init__(
        self,
        partial_dataloader: partial[DataLoader[torch.Tensor]],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
        logger: StepIntervalLogger,
        max_epochs: int = 1,
        minimum_dataset_size: int = 2,
        minimum_new_data_count: int = 0,
        gradient_clip_norm: float | None = None,
    ) -> None:
        """Initialize the MultimodalTemporalEncoderTrainer.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            logger: Logger for tracking training metrics.
            max_epochs: Maximum number of epochs to train for each call to train().
            minimum_dataset_size: Minimum size of dataset required to start training.
            minimum_new_data_count: Minimum number of new data points required to trigger training.
            gradient_clip_norm: Maximum norm for gradient clipping (None to disable).
        """
        super().__init__()
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.device = device
        self.logger = logger
        self.max_epochs = max_epochs
        assert minimum_dataset_size >= 2, "minimum_dataset_size must be at least 2"
        self.minimum_dataset_size = minimum_dataset_size
        self.minimum_new_data_count = minimum_new_data_count
        self.gradient_clip_norm = gradient_clip_norm
        self.dataset_previous_get_time = float("-inf")

    def on_data_users_dict_attached(self) -> None:
        """Set up data user for multimodal temporal data."""
        self.temporal_data_user: ThreadSafeDataUser[MultimodalTemporalDataBuffer] = self.get_data_user(
            BufferNames.MULTIMODAL_TEMPORAL
        )

    def on_model_wrappers_dict_attached(self) -> None:
        """Set up temporal encoder model and optimizer state."""
        self.temporal_encoder: ModelWrapper[MultimodalTemporalEncoder] = self.get_training_model(
            ModelNames.MULTIMODAL_TEMPORAL_ENCODER
        )
        self.optimizer_state = self.partial_optimizer(self.temporal_encoder.parameters()).state_dict()

    def is_trainable(self) -> bool:
        """Check if training can proceed based on data availability."""
        self.temporal_data_user.update()
        return len(self.temporal_data_user.buffer) >= self.minimum_dataset_size and self._is_new_data_available()

    def _is_new_data_available(self) -> bool:
        """Check if enough new data has been added since last training."""
        return (
            self.temporal_data_user.buffer.count_data_added_since(self.dataset_previous_get_time)
            >= self.minimum_new_data_count
        )

    def get_dataset(self) -> Dataset[Any]:
        """Get the current dataset and update the last access time."""
        dataset = self.temporal_data_user.get_dataset()
        self.dataset_previous_get_time = time.time()
        return dataset

    def train(self) -> None:
        """Train the temporal encoder model on multimodal data."""
        self.temporal_encoder.to(self.device)

        optimizer = self.partial_optimizer(self.temporal_encoder.parameters())
        optimizer.load_state_dict(self.optimizer_state)

        dataloader = self.partial_dataloader(dataset=self.get_dataset())

        for _ in range(self.max_epochs):
            for batch in dataloader:
                observations, hiddens = batch

                # Move data to device
                observations = {k: v.to(self.device) for k, v in observations.items()}
                hiddens = hiddens.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                obs_hat_dists: dict[str, Distribution]
                _, _, obs_hat_dists = self.temporal_encoder(observations, hiddens)

                # Calculate losses for each modality
                total_loss = torch.tensor(0.0, device=self.device)
                for modality, dist in obs_hat_dists.items():
                    modal_loss = -dist.log_prob(observations[modality]).mean()
                    self.logger.log(f"temporal_encoder/losses/{modality}_loss", modal_loss)
                    total_loss += modal_loss

                self.logger.log("temporal_encoder/losses/total_loss", total_loss)

                # Backward pass
                total_loss.backward()

                # Log gradient norm
                if self.temporal_encoder.parameters():
                    grad_norm = torch.cat(
                        [p.grad.flatten() for p in self.temporal_encoder.parameters() if p.grad is not None]
                    ).norm()
                    self.logger.log("temporal_encoder/metrics/grad_norm", grad_norm)

                # Gradient clipping if enabled
                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.temporal_encoder.parameters(), self.gradient_clip_norm, error_if_nonfinite=True
                    )

                optimizer.step()
                self.logger.update()

        self.optimizer_state = optimizer.state_dict()

    @override
    def save_state(self, path: Path) -> None:
        """Save trainer state to disk.

        Args:
            path: Directory path to save the state files.
        """
        path.mkdir()
        torch.save(self.optimizer_state, path / "optimizer.pt")
        torch.save(self.logger.state_dict(), path / "logger.pt")
        torch.save(self.dataset_previous_get_time, path / "dataset_previous_get_time.pt")

    @override
    def load_state(self, path: Path) -> None:
        """Load trainer state from disk.

        Args:
            path: Directory path containing the state files.
        """
        self.optimizer_state = torch.load(path / "optimizer.pt")
        self.logger.load_state_dict(torch.load(path / "logger.pt"))
        self.dataset_previous_get_time = torch.load(path / "dataset_previous_get_time.pt")
