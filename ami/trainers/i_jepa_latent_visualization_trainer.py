import time
from functools import partial
from pathlib import Path
from typing import Literal

# import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional
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
from ami.utils import min_max_normalize

from .base_trainer import BaseTrainer


class IJEPALatentVisualizationDecoderTrainer(BaseTrainer):
    """Trainer for IJEPA Decoder."""

    @override
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[Tensor]],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
        logger: StepIntervalLogger,
        decoder_name: Literal[
            ModelNames.I_JEPA_CONTEXT_VISUALIZATION_DECODER, ModelNames.I_JEPA_TARGET_VISUALIZATION_DECODER
        ],
        max_epochs: int = 1,
        minimum_dataset_size: int = 1,
        minimum_new_data_count: int = 0,
        validation_dataloader: DataLoader[tuple[Tensor]] | None = None,
        num_visualize_images: int = 64,
        visualize_grid_row: int = 8,
    ) -> None:
        """Initializes an IJEPALatentVisualizationDecoderTrainer object.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            logger: The logger object for recording training metrics and visualizations.
            decoder_name: Name of the decoder (context or target) to be trained for visualization.
            max_epochs: Maximum number of epochs to train the decoder. Default is 1.
            minimum_dataset_size: Minimum number of samples required in the dataset to start training. Default is 1.
            minimum_new_data_count: Minimum number of new data samples required to run the training. Default is 0.
            validation_dataloader DataLoader instance for validation.
            num_visualize_images: Number of images to use for visualization. Default is 64.
            visualize_grid_row: Number of images per row in the visualization grid. Default is 8.
        """
        super().__init__()

        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.device = device
        self.logger = logger
        self.log_prefix = "i-jepa-latent-visualization/"

        # Prepare self.log_prefix correspond to decoder name.
        match decoder_name:
            case ModelNames.I_JEPA_CONTEXT_VISUALIZATION_DECODER:
                encoder_name = ModelNames.I_JEPA_CONTEXT_ENCODER
                self.log_prefix += "context/"
            case ModelNames.I_JEPA_TARGET_VISUALIZATION_DECODER:
                encoder_name = ModelNames.I_JEPA_TARGET_ENCODER
                self.log_prefix += "target/"
            case _:
                raise ValueError(f"Unexpected decoder_name: {decoder_name}")

        self.encoder_name = encoder_name
        self.decoder_name = decoder_name

        self.max_epochs = max_epochs
        self.minimum_dataset_size = minimum_dataset_size
        self.minimum_new_data_count = minimum_new_data_count
        self.validation_dataloader = validation_dataloader
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

    @torch.inference_mode()
    def validation(self, dataloader: DataLoader[tuple[Tensor]]) -> None:
        """Compute the reconstruction loss and log visualization grid image."""

        input_image_batch_list = []
        reconstruction_image_batch_list = []
        loss_list = []
        batch: tuple[Tensor]
        for batch in dataloader:
            (image_batch,) = batch
            input_image_batch_list.append(image_batch)
            image_batch = image_batch.to(self.device)

            latents = self.encoder.infer(image_batch)
            reconstructions: Tensor = self.decoder(latents)
            rec_img_size = reconstructions.shape[-2:]
            resized_image_batch = torchvision.transforms.v2.functional.resize(image_batch, rec_img_size)
            loss_list.append(F.mse_loss(resized_image_batch, reconstructions, reduction="none").flatten(1).mean(1))

            reconstruction_image_batch_list.append(reconstructions.cpu())

        input_image_batches = torch.cat(input_image_batch_list)
        reconstruction_image_batches = torch.cat(reconstruction_image_batch_list)
        visualize_indices = torch.randperm(input_image_batches.size(0))[: self.num_visualize_images].sort().values
        input_image_selected = input_image_batches[visualize_indices]
        input_image_selected = min_max_normalize(input_image_selected.flatten(1), 0, 1, dim=-1).reshape(
            input_image_selected.shape
        )
        reconstruction_image_selected = reconstruction_image_batches[visualize_indices]
        reconstruction_image_selected = min_max_normalize(
            reconstruction_image_selected.flatten(1), 0, 1, dim=-1
        ).reshape(reconstruction_image_selected.shape)

        grid_input_image = torchvision.utils.make_grid(input_image_selected, self.visualize_grid_row, normalize=True)
        grid_reconstruction_image = torchvision.utils.make_grid(
            reconstruction_image_selected, self.visualize_grid_row, normalize=True
        )
        losses = torch.cat(loss_list)
        loss = torch.mean(losses)

        self.logger.log(self.log_prefix + "losses/validation-reconstruction", loss, force_log=True)
        self.logger.tensorboard.add_image(self.log_prefix + "metrics/input", grid_input_image, self.logger.global_step)
        self.logger.tensorboard.add_image(
            self.log_prefix + "metrics/reconstruction", grid_reconstruction_image, self.logger.global_step
        )

        # fig = plt.figure(figsize=(6.4, 3.6))
        # ax = fig.subplots()
        # losses = losses.cpu()
        # ax.plot(losses.numpy())
        # ax.set_ylim(min(0, losses.min().item()))
        # ax.set_xlabel("past to future")
        # ax.set_ylabel("loss")
        # ax.set_title("losses past to future")
        # self.logger.tensorboard.add_figure(
        #     self.log_prefix + "metrics/losses-past-to-future", fig, self.logger.global_step
        # )

    @override
    def train(self) -> None:
        # move to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        optimizer = self.partial_optimizer(self.decoder.parameters())
        optimizer.load_state_dict(self.optimizer_state)

        # prepare about dataset
        dataloader = self.partial_dataloader(dataset=self.get_dataset())

        for _ in range(self.max_epochs):
            batch: tuple[
                Tensor,
            ]
            for batch in dataloader:
                (image_batch,) = batch
                image_batch = image_batch.to(self.device)

                with torch.no_grad():
                    latents = self.encoder.infer(image_batch)
                    # latents: [batch_size, n_patches_height * n_patches_width, latents_dim]

                image_out: Tensor = self.decoder(latents)
                image_size = image_out.size()[-2:]

                image_batch_resized = torchvision.transforms.v2.functional.resize(image_batch, image_size)

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

        if self.validation_dataloader is not None:
            self.validation(self.validation_dataloader)

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
