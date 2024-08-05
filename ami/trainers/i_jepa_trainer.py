# Ref: https://github.com/facebookresearch/ijepa

from functools import partial
from pathlib import Path
import itertools
import copy

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.interfaces import ThreadSafeDataUser
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.i_jepa import (
    IJEPAEncoder,
    IJEPAPredictor,
    repeat_patches_along_with_batch_axis,
    select_patches_by_indices,
)
from ami.tensorboard_loggers import StepIntervalLogger

from .base_trainer import BaseTrainer
from .components.i_jepa_mask_collator import IJEPAMultiBlockMaskCollator


class IJEPATrainer(BaseTrainer):
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[torch.Tensor]],
        partial_optimizer: partial[Optimizer],
        i_jepa_mask_collator: IJEPAMultiBlockMaskCollator,
        device: torch.device,
        logger: StepIntervalLogger,
        kl_coef: float = 1.0,
        max_epochs: int = 1,
        minimum_dataset_size: int = 1,
        minimum_new_data_count: int = 0,
    ) -> None:
        """Initializes an IJEPATrainer object.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer lacking provided parameters.
            i_jepa_mask_collator: A collator to make masks for I-JEPA models.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            kl_coef: The coefficient for balancing KL divergence relative to the reconstruction loss.
            minimum_new_data_count: Minimum number of new data count required to run the training.
        """
        super().__init__()
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.i_jepa_mask_collator = i_jepa_mask_collator
        self.device = device
        self.logger = logger
        self.kl_coef = kl_coef
        self.max_epochs = max_epochs
        self.minimum_dataset_size = minimum_dataset_size
        self.minimum_new_data_count = minimum_new_data_count

    def on_data_users_dict_attached(self) -> None:
        self.image_data_user: ThreadSafeDataUser[RandomDataBuffer] = self.get_data_user(
            BufferNames.IMAGE
        )

    def on_model_wrappers_dict_attached(self) -> None:
        self.context_encoder: ModelWrapper[IJEPAEncoder] = self.get_training_model(
            "i_jepa_context_encoder"
        )
        self.predictor: ModelWrapper[IJEPAPredictor] = self.get_training_model(
            "i_jepa_predictor"
        )
        self.target_encoder: ModelWrapper[IJEPAEncoder] = self.get_training_model(
            "i_jepa_target_encoder"
        )

        # Since the model is swapped between the inference and training threads each time it is trained,
        # the model and optimizer are built within the `train()` method.
        # The following is the initial state generation of the optimizer.
        self.optimizer_state = self.partial_optimizer(
            itertools.chain(self.context_encoder.parameters(), self.predictor.parameters())
        ).state_dict()

    def is_trainable(self) -> bool:
        self.image_data_user.update()
        return (
            len(self.image_data_user.buffer) >= self.minimum_dataset_size
            and self._is_new_data_available()
        )

    def _is_new_data_available(self) -> bool:
        return self.image_data_user.buffer.new_data_count >= self.minimum_new_data_count

    def train(self) -> None:
        # define models
        context_encoder = self.context_encoder.model
        predictor = self.predictor.model
        target_encoder = self.target_encoder.model
        # move to device
        context_encoder = context_encoder.to(self.device)
        predictor = predictor.to(self.device)
        target_encoder = target_encoder.to(self.device)
        # define optimizer
        optimizer = self.partial_optimizer(
            itertools.chain(context_encoder.parameters(), predictor.parameters())
        )
        optimizer.load_state_dict(self.optimizer_state)
        # prepare about dataset
        dataset = self.image_data_user.get_dataset()
        dataloader = self.partial_dataloader(dataset=dataset, collate_fn=self.i_jepa_mask_collator)

        for _ in range(self.max_epochs):
            for batch in dataloader:
                (image_batch, masks_for_context_encoder, masks_for_predictor) = batch
                assert len(image_batch)==1# must be fixed in another PR
                image_batch = image_batch[0]# must be fixed in another PR
                image_batch = image_batch.to(self.device)
                masks_for_context_encoder = [masks.to(self.device) for masks in masks_for_context_encoder]
                masks_for_predictor = [masks.to(self.device) for masks in masks_for_predictor]
                optimizer.zero_grad()

                # target encoder
                with torch.no_grad():
                    latent_from_target_encoder = target_encoder(image_batch)
                    latent_from_target_encoder = torch.nn.functional.layer_norm(
                        latent_from_target_encoder,
                        (latent_from_target_encoder.size(-1),),
                    )  # normalize over feature-dim
                    batch_size = len(image_batch)
                    # -- create targets (masked regions of h)
                    latent_from_target_encoder = select_patches_by_indices(
                        latent_from_target_encoder, masks_for_predictor
                    )
                    latent_from_target_encoder = repeat_patches_along_with_batch_axis(
                        latent_from_target_encoder,
                        batch_size,
                        len(masks_for_context_encoder),
                    )
                # context encoder
                latent_from_context_encoder = context_encoder(
                    images=image_batch, 
                    patch_selections_for_context_encoder=masks_for_context_encoder
                )
                # predictor
                latent_from_predictor = predictor(
                    latents=latent_from_context_encoder,
                    patch_selections_for_context_encoder=masks_for_context_encoder,
                    patch_selections_for_predictor=masks_for_predictor,
                )
                # calc loss
                loss = torch.nn.functional.smooth_l1_loss(
                    latent_from_predictor,
                    latent_from_target_encoder,
                    reduction="mean",
                )
                self.logger.log("i-jepa/loss", loss)
                loss.backward()
                optimizer.step()
                self.logger.update()
                # target_encoder updates weights by moving average from context_encoder
                with torch.no_grad():
                    # In the original I-JEPA, m changes through training process.
                    # But in ami-q, since assuming Semi-permanent training, m is set as fixed value.
                    m = 0.996 # based on the original I-JEPA initinal setting.
                    for target_encoder_param, context_encoder_param in zip(
                        target_encoder.parameters(), context_encoder.parameters()
                    ):
                        target_encoder_param.data.mul_(m).add_(
                            (1.0 - m) * context_encoder_param.detach().data
                        )

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
