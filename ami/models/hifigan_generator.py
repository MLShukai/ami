# Ref:
# https://github.com/jik876/hifi-gan/blob/master/models.py

import logging

import torch
from torch.nn import Conv1d, ConvTranspose1d

logger = logging.getLogger(__name__)


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if isinstance(m, Conv1d | ConvTranspose1d):
        m.weight.data.normal_(mean, std)


# Avoiding changing the size of inputs and outputs of Conv1ds in ResBlock1.
def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilations: list[int] = [1, 3, 5]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for dilation in dilations:
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.utils.parametrizations.weight_norm(
                        Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=dilation,
                            padding=get_padding(kernel_size, dilation),
                        )
                    ),
                    torch.nn.LeakyReLU(0.1),
                    torch.nn.utils.parametrizations.weight_norm(
                        Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=1,
                            padding=get_padding(kernel_size, 1),
                        )
                    ),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x) + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            torch.nn.utils.parametrize.remove_parametrizations(layer)
        for layer in self.convs2:
            torch.nn.utils.parametrize.remove_parametrizations(layer)


class HifiGANGenerator(torch.nn.Module):
    """Generate waveforms from input features such as spectrogram, latents, and
    so on."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        upsample_rates: list[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: list[int] = [16, 16, 4, 4],
        upsample_paddings: list[int] = [
            4,
            4,
            1,
            1,
        ],  # Parameters not present in the original implementation. Calculated with (upsample_kernel_size - upsample_rate) // 2
        upsample_initial_channel: int = 128,
        resblock_kernel_sizes: list[int] = [3, 7, 11],
        resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = torch.nn.utils.parametrizations.weight_norm(
            Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        )

        self.layers = torch.nn.ModuleList()
        for i, (rate, kernel_size, padding) in enumerate(zip(upsample_rates, upsample_kernel_sizes, upsample_paddings)):
            self.layers.append(
                torch.nn.utils.parametrizations.weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size,
                        rate,
                        padding=padding,
                    )
                )
            )

        self.resblocks = torch.nn.ModuleList()
        for i in range(len(self.layers)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation_size in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, kernel_size, dilation_size))

        self.conv_post = torch.nn.utils.parametrizations.weight_norm(Conv1d(ch, out_channels, 7, 1, padding=3))
        self.layers.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate waveforms from input features such as spectrogram, latents,
        and so on.

        Args:
            features (Tensor):
                Input features. Shape: [batch_size, in_channels, frames]

        Returns:
            Tensor:
                Generated waveforms. Shape: [batch_size, out_channels, sample_size]
        """
        x = self.conv_pre(features)
        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, 0.1)
            x = self.layers[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            assert isinstance(xs, torch.Tensor)
            x = xs / self.num_kernels
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self) -> None:
        logger("Removing weight norm...")
        for layer in self.layers:
            torch.nn.utils.parametrize.remove_parametrizations(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        torch.nn.utils.parametrize.remove_parametrizations(self.conv_pre)
        torch.nn.utils.parametrize.remove_parametrizations(self.conv_post)
