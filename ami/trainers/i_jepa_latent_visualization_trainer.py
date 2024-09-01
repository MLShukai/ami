import time
from functools import partial
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as torchvisF
import torchvision.utils
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.interfaces import ThreadSafeDataUser
from ami.models.bool_mask_i_jepa import BoolMaskIJEPAEncoder
from ami.models.i_jepa import IJEPAEncoder
from ami.models.i_jepa_latent_visualization_decoder import (
    IJEPALatentVisualizationDecoder,
)
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.tensorboard_loggers import StepIntervalLogger

from .base_trainer import BaseTrainer


class IJEPALatentVisualizationTrainer(BaseTrainer):
    """Trainer for IJEPA Decoder."""

    @override
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[Tensor]],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
        logger: StepIntervalLogger,
        encoder_name: Literal[ModelNames.I_JEPA_CONTEXT_ENCODER, ModelNames.I_JEPA_TARGET_ENCODER],
        decoder_name: Literal[ModelNames.I_JEPA_CONTEXT_DECODER, ModelNames.I_JEPA_TARGET_DECODER],
        max_epochs: int = 1,
        minimum_dataset_size: int = 1,
        minimum_new_data_count: int = 0,
        num_visualize_images: int = 64,
        visualize_grid_row: int = 8,
    ) -> None:
        """Initializes an IJEPALatentVisualizationDecoderTrainer object.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            encoder_type: Encoder to be visualized.
            minimum_new_data_count: Minimum number of new data count required to run the training.
        """
        super().__init__()

        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.device = device
        self.logger = logger
        self.log_prefix = "i-jepa-latent-visualization"

        # Checks the decoder name correspond to encoder name.
        match encoder_name:
            case ModelNames.I_JEPA_CONTEXT_ENCODER:
                assert decoder_name == ModelNames.I_JEPA_CONTEXT_DECODER, "<message>"
                self.log_prefix += " (context)"
            case ModelNames.I_JEPA_TARGET_ENCODER:
                assert decoder_name == ModelNames.I_JEPA_TARGET_DECODER, "<message>"
                self.log_prefix += " (target)"
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name

        self.max_epochs = max_epochs
        self.minimum_dataset_size = minimum_dataset_size
        self.minimum_new_data_count = minimum_new_data_count
        self.num_visualize_images = num_visualize_images
        self.visualize_grid_row = visualize_grid_row

        self.dataset_previous_get_time = float("-inf")

    def on_data_users_dict_attached(self) -> None:
        self.image_data_user: ThreadSafeDataUser[RandomDataBuffer] = self.get_data_user(BufferNames.IMAGE)

    def on_model_wrappers_dict_attached(self) -> None:
        self.encoder: ModelWrapper[BoolMaskIJEPAEncoder | IJEPAEncoder] = self.get_frozen_model(self.encoder_name)
        self.decoder: ModelWrapper[IJEPALatentVisualizationDecoder] = self.get_training_model(self.decoder_name)

        self.optimizer_state = self.partial_optimizer(self.decoder.parameters()).state_dict()

    @override
    def is_trainable(self) -> bool:
        self.image_data_user.update()
        return len(self.image_data_user.buffer) >= self.minimum_dataset_size and self._is_new_data_available()

    def _is_new_data_available(self) -> bool:
        return (
            self.image_data_user.buffer.count_data_added_since(self.dataset_previous_get_time)
            >= self.minimum_new_data_count
        )

    def get_dataset(self) -> Dataset[Tensor]:
        dataset = self.image_data_user.get_dataset()
        self.dataset_previous_get_time = time.time()
        return dataset

    @torch.no_grad()
    def make_reconstruction_image_grid(self, dataloader: DataLoader[Tensor]) -> Tensor:
        """Makes the reconstruction image grid."""
        reconstruction_image_batches = []
        batch: tuple[Tensor]
        num_remaining = self.num_visualize_images
        for batch in dataloader:
            (image_batch,) = batch
            image_batch = image_batch.to(self.device)[:num_remaining]

            latents = self.encoder(image_batch)
            reconstruction: Tensor = self.decoder(latents)
            reconstruction_image_batches.append(reconstruction.cpu())
            num_remaining -= reconstruction.size(0)
            if num_remaining <= 0:
                break
        reconstruction_images = torch.cat(reconstruction_image_batches)
        return torchvision.utils.make_grid(reconstruction_images, self.visualize_grid_row)

    @override
    def train(self) -> None:
        # move to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        optimizer = self.partial_optimizer(self.decoder.parameters())
        optimizer.load_state_dict(self.optimizer_state)

        # prepare about dataset
        dataloader = self.partial_dataloader(dataset=self.get_dataset())

        # for logging

        for _ in range(self.max_epochs):
            batch: tuple[
                Tensor,
            ]
            for batch in dataloader:
                (image_batch,) = batch
                image_batch = image_batch.to(self.device)

                with torch.no_grad():
                    latents = self.encoder(image_batch)
                    # latents: [batch_size, n_patches_height * n_patches_width, latents_dim]

                image_out: Tensor = self.decoder(latents)
                image_size = image_out.size()[-2:]

                image_batch_resized = torchvisF.resize(image_batch, image_size)

                # calc loss
                loss = F.mse_loss(
                    image_out,
                    image_batch_resized,
                    reduction="mean",
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.logger.log(self.log_prefix + "losses/reconstruction", loss)

                self.logger.update()

        reconstruction = self.make_reconstruction_image_grid(dataloader)
        self.logger.tensorboard.add_image(
            self.log_prefix + "metrics/reconstruction", reconstruction, self.logger.global_step
        )

        self.optimizer_state = optimizer.state_dict()
        self.logger_state = self.logger.state_dict()

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
