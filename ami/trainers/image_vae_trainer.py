from functools import partial

import torch
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ami.data.buffers.buffer_names import BufferNames
from ami.models.model_names import ModelNames
from ami.models.vae import VAE, Conv2dDecoder, Conv2dEncoder

from .base_trainer import BaseTrainer


class ImageVAETrainer(BaseTrainer):
    def __init__(
        self,
        partial_dataloader: partial[DataLoader[torch.Tensor]],
        partial_optimizer: partial[Optimizer],
        device: torch.device,
    ):
        super().__init__()
        self.partial_optimizer = partial_optimizer
        self.partial_dataloader = partial_dataloader
        self.device = device

    def on_data_users_dict_attached(self) -> None:
        self.data_user = self.get_data_user(BufferNames.IMAGE_BUFFER)

    def on_model_wrappers_dict_attached(self) -> None:
        self.encoder: Conv2dEncoder = self.get_training_model(ModelNames.IMAGE_ENCODER).model
        self.decoder: Conv2dDecoder = self.get_training_model(ModelNames.IMAGE_DECODER).model
        self.vae = VAE(self.encoder, self.decoder)
        self.optimizer_state = self.partial_optimizer(self.vae.parameters()).state_dict()

    def train(self) -> None:
        optimizer = self.partial_optimizer(self.vae.parameters())
        optimizer.load_state_dict(self.optimizer_state)
        dataset = self.data_user.get_new_dataset()
        dataloader = self.partial_dataloader(dataset=dataset)
        for batch in dataloader:
            (image_batch,) = batch
            image_batch = image_batch.to(self.device)
            optimizer.zero_grad()
            image_batch_reconstructed, dist_batch = self.vae(image_batch)
            rec_loss = mse_loss(image_batch, image_batch_reconstructed)
            kl_loss = kl_divergence(
                dist_batch, Normal(torch.zeros_like(dist_batch.mean), torch.ones_like(dist_batch.stddev))
            )
            loss = rec_loss + kl_loss.sum()
            loss.backward()
            optimizer.step()
        self.optimizer_state = self.partial_optimizer(self.vae.parameters()).state_dict()
