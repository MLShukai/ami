# Ref:
# https://github.com/jik876/hifi-gan/blob/master/models.py

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm, spectral_norm

def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)

class DiscriminatorP(nn.Module):
    def __init__(
        self, 
        period: int, 
        in_channels: int=1, 
        kernel_size: int=5, 
        stride: int=3,
    ) -> None:
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(in_channels, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, input_waveforms: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmaps: list[torch.Tesnor] = []
        x = input_waveforms
        
        # 1d to 2d
        batch_size, channels, sample_size = x.shape
        if sample_size % self.period != 0: # pad first
            n_pad = self.period - (sample_size % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            sample_size = sample_size + n_pad
        x = x.view(batch_size, channels, sample_size // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmaps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, in_channels: int=1) -> None:
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(period=2, in_channels=in_channels),
            DiscriminatorP(period=3, in_channels=in_channels),
            DiscriminatorP(period=5, in_channels=in_channels),
            DiscriminatorP(period=7, in_channels=in_channels),
            DiscriminatorP(period=11, in_channels=in_channels),
        ])

    def forward(
        self, real_waveforms: torch.Tensor, fake_waveforms: torch.Tensor
    ) -> tuple[
        list[torch.Tensor], 
        list[torch.Tensor],
        list[list[torch.Tensor]], 
        list[list[torch.Tensor]],
    ]:
        y_d_rs: list[torch.Tensor] = []
        y_d_fs: list[torch.Tensor] = []
        fmaps_rs: list[list[torch.Tensor]] = []
        fmaps_fs: list[list[torch.Tensor]] = []
        for discriminator in self.discriminators:
            y_d_r, fmaps_r = discriminator(real_waveforms)
            y_d_g, fmaps_g = discriminator(fake_waveforms)
            y_d_rs.append(y_d_r)
            y_d_fs.append(y_d_g)
            fmaps_rs.append(fmaps_r)
            fmaps_fs.append(fmaps_g)
        return y_d_rs, y_d_fs, fmaps_rs, fmaps_fs


class DiscriminatorS(nn.Module):
    def __init__(self, in_channels: int=1, use_spectral_norm: bool=False) -> None:
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(in_channels, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, input_waveforms: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
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
    def __init__(self, in_channels: int=1) -> None:
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(in_channels=in_channels, use_spectral_norm=True),
            DiscriminatorS(in_channels=in_channels),
            DiscriminatorS(in_channels=in_channels),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(
        self, real_waveforms: torch.Tensor, fake_waveforms: torch.Tensor
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[torch.Tensor]],
        list[list[torch.Tensor]],
    ]:
        y = real_waveforms
        y_hat = fake_waveforms
        y_d_rs: list[torch.Tensor] = []
        y_d_fs: list[torch.Tensor] = []
        fmaps_rs: list[list[torch.Tensor]] = []
        fmaps_fs: list[list[torch.Tensor]] = []
        for i, discriminator in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmaps_r = discriminator(y)
            y_d_g, fmaps_g = discriminator(y_hat)
            y_d_rs.append(y_d_r)
            fmaps_rs.append(fmaps_r)
            y_d_fs.append(y_d_g)
            fmaps_fs.append(fmaps_g)
        return y_d_rs, y_d_fs, fmaps_rs, fmaps_fs