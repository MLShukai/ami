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
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.interfaces import ThreadSafeDataUser
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.vae import VAE, Decoder, Encoder
from ami.tensorboard_loggers import StepIntervalLogger

from .base_trainer import BaseTrainer


class ImageVAETrainer(BaseTrainer):
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[torch.Tensor]],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
        logger: StepIntervalLogger,
        kl_coef: float = 1.0,
        max_epochs: int = 1,
        minimum_dataset_size: int = 1,
        minimum_new_data_count: int = 0,
    ) -> None:
        """Initializes an ImageVAETrainer object.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            kl_coef: The coefficient for balancing KL divergence relative to the reconstruction loss.
            minimum_new_data_count: Minimum number of new data count required to run the training.
        """
        super().__init__()
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.device = device
        self.logger = logger
        self.kl_coef = kl_coef
        self.max_epochs = max_epochs
        self.minimum_dataset_size = minimum_dataset_size
        self.minimum_new_data_count = minimum_new_data_count

    def on_data_users_dict_attached(self) -> None:
        self.image_data_user: ThreadSafeDataUser[RandomDataBuffer] = self.get_data_user(BufferNames.IMAGE)

    def on_model_wrappers_dict_attached(self) -> None:
        self.encoder: ModelWrapper[Encoder] = self.get_training_model(ModelNames.IMAGE_ENCODER)
        self.decoder: ModelWrapper[Decoder] = self.get_training_model(ModelNames.IMAGE_DECODER)

        # モデルは学習する度に推論スレッドと学習スレッドで入れ替えられるため、VAEとオプティマイザは`train()`メソッド内で構築する。
        # 下記ではオプティマイザの初期状態生成を行う。
        vae = VAE(self.encoder.model, self.decoder.model)
        self.optimizer_state = self.partial_optimizer(vae.parameters()).state_dict()

    def is_trainable(self) -> bool:
        self.image_data_user.update()
        return len(self.image_data_user.buffer) >= self.minimum_dataset_size and self._is_new_data_available()

    def _is_new_data_available(self) -> bool:
        return self.image_data_user.buffer.new_data_count >= self.minimum_new_data_count

    def train(self) -> None:
        vae = VAE(self.encoder.model, self.decoder.model)
        vae.to(self.device)

        optimizer = self.partial_optimizer(vae.parameters())
        optimizer.load_state_dict(self.optimizer_state)
        dataset = self.image_data_user.get_dataset()
        dataloader = self.partial_dataloader(dataset=dataset)

        for _ in range(self.max_epochs):
            for batch in dataloader:
                (image_batch,) = batch
                image_batch = image_batch.to(self.device)
                optimizer.zero_grad()
                image_batch_reconstructed, dist_batch = vae(image_batch)
                rec_loss = mse_loss(image_batch, image_batch_reconstructed)
                kl_loss = kl_divergence(
                    dist_batch, Normal(torch.zeros_like(dist_batch.mean), torch.ones_like(dist_batch.stddev))
                ).mean()
                loss = rec_loss + self.kl_coef * kl_loss
                self.logger.log("image_vae/loss", loss)
                self.logger.log("image_vae/reconstruction_loss", rec_loss)
                self.logger.log("image_vae/kl_loss", kl_loss)
                loss.backward()
                optimizer.step()
                self.logger.update()

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
