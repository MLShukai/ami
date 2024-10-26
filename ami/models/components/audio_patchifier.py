# Ref: https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py#L844

import dataclasses

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)
        self.layer_norm = nn.LayerNorm(out_channels, elementwise_affine=True)
        self.non_linear_layer = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.transpose(-2, -1)
        x = self.layer_norm(x)
        x = x.transpose(-2, -1)
        x = self.non_linear_layer(x)
        return x


class AudioPatchifier(nn.Module):
    """Convert input audios into patch embeddings."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 512,
    ) -> None:
        """

        Args:
            in_channels (int):
                Num of input audio channels.
            embed_dim (int):
                Num of embed dimensions per a patch
                Defaults to 512.
        """
        super().__init__()

        # According to data2vec paper (https://arxiv.org/abs/2202.03555),
        # 1D Convolutions with strides (5,2,2,2,2,2,2) and kernel widths (10,3,3,3,3,2,2)
        # results in window_size=400[samples] and hop_size=320[samples] as total.
        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=embed_dim, kernel_size=10, stride=5),
            ConvBlock(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=2),
            ConvBlock(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=2),
            ConvBlock(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=2),
            ConvBlock(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=2),
            ConvBlock(in_channels=embed_dim, out_channels=embed_dim, kernel_size=2, stride=2),
            ConvBlock(in_channels=embed_dim, out_channels=embed_dim, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input audios into patches.

        Args:
            x (torch.Tensor):
                Input audios. Shape is [batch_size, n_channels, n_samples].

        Returns:
            torch.Tensor:
                Output patches. Shape is [batch_size, n_patches, embed_dim].
        """
        x = self.conv_layers(x)
        # [batch_size, embed_dim, n_patches] -> [batch_size, n_patches, embed_dim]
        x = x.transpose(-1, -2)
        return x
