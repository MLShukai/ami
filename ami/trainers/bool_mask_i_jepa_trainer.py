import itertools
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.interfaces import ThreadSafeDataUser
from ami.models.bool_mask_i_jepa import BoolMaskIJEPAEncoder, BoolMaskIJEPAPredictor
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.tensorboard_loggers import StepIntervalLogger

from .base_trainer import BaseTrainer


class BoolMaskIJEPATrainer(BaseTrainer):
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[torch.Tensor]],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
        logger: StepIntervalLogger,
        target_encoder_update_moving_average: float = 0.996,  # based on the original I-JEPA initinal setting.
        max_epochs: int = 1,
        minimum_dataset_size: int = 1,
        minimum_new_data_count: int = 0,
    ) -> None:
        """Initializes an BoolMaskIJEPATrainer object.

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
        self.target_encoder_update_moving_average = target_encoder_update_moving_average
        self.max_epochs = max_epochs
        self.minimum_dataset_size = minimum_dataset_size
        self.minimum_new_data_count = minimum_new_data_count

    def on_data_users_dict_attached(self) -> None:
        self.image_data_user: ThreadSafeDataUser[RandomDataBuffer] = self.get_data_user(BufferNames.IMAGE)

    def on_model_wrappers_dict_attached(self) -> None:
        self.context_encoder: ModelWrapper[BoolMaskIJEPAEncoder] = self.get_training_model(
            ModelNames.I_JEPA_CONTEXT_ENCODER
        )
        self.predictor: ModelWrapper[BoolMaskIJEPAPredictor] = self.get_training_model(ModelNames.I_JEPA_PREDICTOR)
        self.target_encoder: ModelWrapper[BoolMaskIJEPAEncoder] = self.get_training_model(
            ModelNames.I_JEPA_TARGET_ENCODER
        )
        assert (
            self.context_encoder.model is not self.target_encoder.model
        ), "context_encoder and target_encoder must be allocated in memory as separate entities."

        # Since the model is swapped between the inference and training threads each time it is trained,
        # the model and optimizer are built within the `train()` method.
        # The following is the initial state generation of the optimizer.
        self.optimizer_state = self.partial_optimizer(
            itertools.chain(self.context_encoder.parameters(), self.predictor.parameters())
        ).state_dict()

        # copy weights from target_encoder to context_encoder
        self.context_encoder.load_state_dict(self.target_encoder.state_dict())

    def is_trainable(self) -> bool:
        self.image_data_user.update()
        return len(self.image_data_user.buffer) >= self.minimum_dataset_size and self._is_new_data_available()

    def _is_new_data_available(self) -> bool:
        return self.image_data_user.buffer.new_data_count >= self.minimum_new_data_count

    def train(self) -> None:
        # move to device
        self.context_encoder = self.context_encoder.to(self.device)
        self.predictor = self.predictor.to(self.device)
        self.target_encoder = self.target_encoder.to(self.device)
        # define optimizer
        optimizer = self.partial_optimizer(
            itertools.chain(self.context_encoder.parameters(), self.predictor.parameters())
        )
        optimizer.load_state_dict(self.optimizer_state)
        # prepare about dataset
        dataset = self.image_data_user.get_dataset()
        dataloader = self.partial_dataloader(dataset=dataset)

        for _ in range(self.max_epochs):
            batch: tuple[Tensor, Tensor, Tensor]
            for batch in dataloader:
                (image_batch, masks_for_context_encoder, masks_for_predictor) = batch
                image_batch = image_batch.to(self.device)
                masks_for_context_encoder = masks_for_context_encoder.to(self.device)
                masks_for_predictor = masks_for_predictor.to(self.device)

                # target encoder
                with torch.no_grad():
                    latent_from_target_encoder: Tensor = self.target_encoder(image_batch)
                    # normalize over feature-dim
                    latent_from_target_encoder = F.layer_norm(
                        latent_from_target_encoder, (latent_from_target_encoder.size(-1),)
                    )

                # context encoder
                latent_from_context_encoder = self.context_encoder(image_batch, masks_for_context_encoder)

                # predictor
                latent_from_predictor = self.predictor(latent_from_context_encoder, masks_for_predictor)

                # Element wise smooth l1 loss for masking.
                losses = F.smooth_l1_loss(latent_from_predictor, latent_from_target_encoder, reduction="none").mean(-1)
                # shape: [batch, n_patches]
                # Apply mask.
                losses = torch.masked_fill(losses, masks_for_predictor, 0.0)
                loss = losses.sum() / masks_for_predictor.logical_not().sum()

                self.logger.log("i-jepa/loss", loss)
                loss.backward()
                optimizer.step()
                self.logger.update()

                # target_encoder updates weights by moving average from context_encoder
                with torch.no_grad():
                    # In the original I-JEPA, m changes through training process.
                    # But in ami-q, since assuming Semi-permanent training, m is set as fixed value.
                    m = self.target_encoder_update_moving_average
                    for target_encoder_param, context_encoder_param in zip(
                        self.target_encoder.parameters(), self.context_encoder.parameters()
                    ):
                        target_encoder_param.data.mul_(m).add_((1.0 - m) * context_encoder_param.detach().data)

        self.optimizer_state = optimizer.state_dict()
        self.logger_state = self.logger.state_dict()

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        torch.save(self.optimizer_state, path / "optimizer.pt")
        torch.save(self.logger.state_dict(), path / "logger.pt")

    @override
    def load_state(self, path: Path) -> None:
        self.optimizer_state = torch.load(path / "optimizer.pt")
        self.logger.load_state_dict(torch.load(path / "logger.pt"))
