# Ref:
# https://github.com/jik876/hifi-gan/blob/master/models.py

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class PeriodDiscriminator(nn.Module):
    """PeriodDiscriminator that constitutes MultiPeriodDiscriminator described
    below.

    Discriminate authenticity of the periodicity of the input audio.
    """

    def __init__(
        self,
        period: int,
        in_channels: int = 1,
        kernel_size: int = 5,
        stride: int = 3,
    ) -> None:
        super().__init__()
        self.period = period

        channels = [
            (in_channels, 32),
            (32, 128),
            (128, 512),
            (512, 1024),
            (1024, 1024),
        ]
        self.convs = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(channels, start=1):
            self.convs.append(
                weight_norm(
                    nn.Conv1d(
                        in_ch,
                        out_ch,
                        kernel_size,
                        stride=1 if i == len(channels) else stride,
                        padding=2 if i == len(channels) else get_padding(kernel_size, 1),
                    )
                )
            )
        self.conv_post = weight_norm(nn.Conv1d(channels[-1][-1], 1, 3, 1, padding=1))

    def forward(self, input_waveforms: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Discriminate authenticity of the periodicity of the input audio.

        Args:
            input_waveforms (torch.Tensor):
                Input waveforms. Shape: [batch_size, in_channels, sample_size]

        Returns:
            torch.Tensor:
                Authenticities. Shape: [batch_size, frames]
            list[torch.Tensor]:
                Feature maps from layers. Shape of each tensor in list: [batch_size, channels, each_n_frames, self.period]
        """
        fmaps: list[torch.Tensor] = []
        x = input_waveforms

        # 1d to 2d
        batch_size, channels, sample_size = x.shape
        if sample_size % self.period != 0:  # pad first
            n_pad = self.period - (sample_size % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            sample_size = sample_size + n_pad
        x = x.view(batch_size, channels, sample_size // self.period, self.period)
        x = x.movedim(-1, 0)
        x = x.contiguous().view(batch_size * self.period, channels, sample_size // self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap = x.contiguous().view(self.period, batch_size, x.size(-2), x.size(-1))
            fmap = fmap.movedim(0, -1)
            fmaps.append(fmap)
        x = self.conv_post(x)
        x = x.contiguous().view(self.period, batch_size, x.size(-2), x.size(-1))
        x = x.movedim(0, -1)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmaps


class MultiPeriodDiscriminator(nn.Module):
    """Discriminate authenticity of the periodicity of the input audio."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(period=period, in_channels=in_channels) for period in periods]
        )

    def forward(self, input_waveforms: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """Discriminate authenticity of the periodicity of the input audio.

        Args:
            input_waveforms (torch.Tensor):
                Real waveforms selected from datasets or fake waveforms generated from model. Shape: [batch_size, in_channels, sample_size]

        Returns:
            list[torch.Tensor]:
                Authenticities of input waveforms by discriminators.
                Shape of each tensor: [batch_size, frames]
            list[list[torch.Tensor]]:
                Intermediate feature maps.
                Shape of each tensor: [batch_size, channels, each_n_frames, self.period]
        """
        authenticity_list: list[torch.Tensor] = []
        fmaps_list: list[list[torch.Tensor]] = []
        for discriminator in self.discriminators:
            authenticity, fmaps = discriminator(input_waveforms)
            authenticity_list.append(authenticity)
            fmaps_list.append(fmaps)
        return authenticity_list, fmaps_list


class ScaleDiscriminator(nn.Module):
    """ScaleDiscriminator that constitutes MultiScaleDiscriminator described
    below.

    Because each PeriodDiscriminator in MultiPeriodDiscriminator only
    accepts disjoint samples, Adding MultiScaleDiscriminator to
    consecutively evaluate the audio sequence.
    """

    def __init__(self, in_channels: int = 1, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(in_channels, 128, 15, 1, padding=7)),
                norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, input_waveforms: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Discriminate authenticity of the input audio consecutively.

        Args:
            input_waveforms (torch.Tensor):
                Input waveforms. Shape: [batch_size, in_channels, sample_size]

        Returns:
            torch.Tensor:
                Authenticities. Shape: [batch_size, frames]
            list[torch.Tensor]:
                Feature maps from layers. Shape of each tensor in list: [batch_size, channels, each_n_frames]
        """
        fmaps: list[torch.Tensor] = []
        x = input_waveforms
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmaps


class MultiScaleDiscriminator(nn.Module):
    """Because each PeriodDiscriminator in MultiPeriodDiscriminator only
    accepts disjoint samples, Adding MultiScaleDiscriminator to consecutively
    evaluate the audio sequence."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(in_channels=in_channels, use_spectral_norm=True),
                ScaleDiscriminator(in_channels=in_channels),
                ScaleDiscriminator(in_channels=in_channels),
            ]
        )
        self.meanpools = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)

    def forward(self, input_waveforms: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """Discriminate authenticity of the input audio consecutively.

        Args:
            input_waveforms (torch.Tensor):
                Real waveforms selected from datasets or fake waveforms generated from model. Shape: [batch_size, in_channels, sample_size]

        Returns:
            list[torch.Tensor]:
                Authenticities of input waveforms by discriminators.
                Shape of each tensor: [batch_size, frames]
            list[list[torch.Tensor]]:
                Intermediate feature maps.
                Shape of each tensor: [batch_size, channels, each_n_frames, self.period]
        """
        authenticity_list: list[torch.Tensor] = []
        fmaps_list: list[list[torch.Tensor]] = []
        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                input_waveforms = self.meanpools(input_waveforms)
            authenticity, fmaps = discriminator(input_waveforms)
            authenticity_list.append(authenticity)
            fmaps_list.append(fmaps)
        return authenticity_list, fmaps_list
