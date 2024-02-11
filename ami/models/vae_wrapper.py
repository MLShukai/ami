import torch
import torch.nn as nn

from .components.vae import VAE
from .model_wrapper import ModelWrapper


class VAEWrapper(ModelWrapper[nn.Module]):
    def __init__(
        self,
        model: VAE,
        default_device: torch.device = torch.device("cpu"),
        has_inference: bool = True,
    ) -> None:
        super().__init__(model, default_device=default_device, has_inference=has_inference)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encoder(x.to(self.device)).rsample()
