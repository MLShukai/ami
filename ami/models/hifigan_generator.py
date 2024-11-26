# Ref:
# https://github.com/jik876/hifi-gan/blob/master/models.py

import logging

import torch
from torch.nn import Conv1d, ConvTranspose1d

logger = logging.getLogger(__name__)


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if isinstance(m, Conv1d | ConvTransposed1d):
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: list[int] = [1, 3, 5]) -> None:
        super().__init__()
        self.convs1 = torch.nn.ModuleList(
            [
                torch.nn.utils.weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                torch.nn.utils.weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                torch.nn.utils.weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = torch.nn.ModuleList(
            [
                torch.nn.utils.weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                torch.nn.utils.weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                torch.nn.utils.weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = torch.nn.functional.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            torch.nn.utils.remove_weight_norm(layer)


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
        self.conv_pre = torch.nn.utils.weight_norm(Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))

        self.layers = torch.nn.ModuleList()
        for i, (upsample_rate, upsample_kernel_size, upsample_padding) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes, upsample_paddings)
        ):
            self.layers.append(
                torch.nn.utils.weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        upsample_kernel_size,
                        upsample_rate,
                        padding=upsample_padding,
                    )
                )
            )

        self.resblocks = torch.nn.ModuleList()
        for i in range(len(self.layers)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for resblock_kernel_size, resblock_dilation_size in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, resblock_kernel_size, resblock_dilation_size))

        self.conv_post = torch.nn.utils.weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
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
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        torch.nn.utils.remove_weight_norm(self.conv_pre)
        torch.nn.utils.remove_weight_norm(self.conv_post)
