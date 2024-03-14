from functools import partial

import torch
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ami.data.buffers.buffer_names import BufferNames
from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.interfaces import ThreadSafeDataUser
from ami.models.model_names import ModelNames
from ami.models.model_wrapper import ModelWrapper
from ami.models.vae import VAE, Decoder, Encoder

from .base_trainer import BaseTrainer


class ImageVAETrainer(BaseTrainer):
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[torch.Tensor]],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
        kl_coef: float = 1.0,
    ) -> None:
        """Initializes an ImageVAETrainer object.

        Args:
            partial_dataloader: A partially instantiated dataloader lacking a provided dataset.
            partial_optimizer: A partially instantiated optimizer lacking provided parameters.
            device: The accelerator device (e.g., CPU, GPU) utilized for training the model.
            kl_coef: The coefficient for balancing KL divergence relative to the reconstruction loss.
        """
        super().__init__()
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.device = device
        self.kl_coef = kl_coef

    def on_data_users_dict_attached(self) -> None:
        self.image_data_user: ThreadSafeDataUser[RandomDataBuffer] = self.get_data_user(BufferNames.IMAGE)

    def on_model_wrappers_dict_attached(self) -> None:
        self.encoder: ModelWrapper[Encoder] = self.get_training_model(ModelNames.IMAGE_ENCODER)
        self.decoder: ModelWrapper[Decoder] = self.get_training_model(ModelNames.IMAGE_DECODER)

        # モデルは学習する度に推論スレッドと学習スレッドで入れ替えられるため、VAEとオプティマイザは`train()`メソッド内で構築する。
        # 下記ではオプティマイザの初期状態生成を行う。
        vae = VAE(self.encoder.model, self.decoder.model)
        self.optimizer_state = self.partial_optimizer(vae.parameters()).state_dict()

    def train(self) -> None:
        vae = VAE(self.encoder.model, self.decoder.model)
        optimizer = self.partial_optimizer(vae.parameters())
        optimizer.load_state_dict(self.optimizer_state)
        dataset = self.image_data_user.get_dataset()
        dataloader = self.partial_dataloader(dataset=dataset)

        for batch in dataloader:
            (image_batch,) = batch
            image_batch = image_batch.to(self.device)
            optimizer.zero_grad()
            image_batch_reconstructed, dist_batch = vae(image_batch)
            rec_loss = mse_loss(image_batch, image_batch_reconstructed)
            kl_loss = kl_divergence(
                dist_batch, Normal(torch.zeros_like(dist_batch.mean), torch.ones_like(dist_batch.stddev))
            )
            loss = rec_loss + self.kl_coef * kl_loss.mean()
            loss.backward()
            optimizer.step()

        self.optimizer_state = optimizer.state_dict()
