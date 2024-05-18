from functools import partial
from typing import Callable, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_1_t


class GlobalConvolutions1d(nn.Module):
    """1D convolutional network with optional batch normalization and global
    pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: _size_1_t,
        padding: _size_1_t | str = 0,
        do_batchnorm: bool = False,
        global_pool_method: Literal["mean", "max"] = "mean",
    ) -> None:
        """Initializes the GlobalConvolutions1d module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels for the final layer.
            num_layers: Number of convolutional layers.
            kernel_size: Size of the convolutional kernel.
            padding: Padding for convolutional layers.
            do_batchnorm: If True, apply batch normalization.
            global_pool_method: Method for global pooling.
        """
        super().__init__()
        self.global_pool_method = global_pool_method

        self.layers = nn.Sequential()
        inch = in_channels
        outch = in_channels
        for i in range(num_layers):
            if i == num_layers - 1:
                outch = out_channels
            self.layers.append(nn.Conv1d(inch, outch, kernel_size=kernel_size, padding=padding))

            if do_batchnorm:
                self.layers.append(nn.BatchNorm1d(outch))

            self.layers.append(nn.ReLU())

        match global_pool_method:
            case "mean":
                self.pool: Callable[[Tensor], Tensor] = partial(torch.mean, dim=-1)
            case "max":

                def torch_max(t: torch.Tensor) -> torch.Tensor:
                    return t.max(dim=-1).values

                self.pool = torch_max

            case _:
                raise ValueError(f"Unknown global pooling method: '{global_pool_method}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (B, C, L).

        Returns:
            Output tensor after convolutional layers and global pooling.
        """
        out = self.layers(x)
        return self.pool(out)
