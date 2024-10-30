# Ref:
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py#L844
# https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/modalities/audio.py#L29

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
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.transpose(-2, -1)
        x = self.layer_norm(x)
        x = x.transpose(-2, -1)
        x = self.activation(x)
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
        kernel_sizes = (10, 3, 3, 3, 3, 2, 2)
        strides = (5, 2, 2, 2, 2, 2, 2)
        # results in window_size=400[samples] and hop_size=320[samples] as total.

        self.conv_layers = nn.Sequential()
        for i, (k, s) in enumerate(zip(kernel_sizes, strides)):
            self.conv_layers.append(
                ConvBlock(
                    in_channels=in_channels if i == 0 else embed_dim,
                    out_channels=embed_dim,
                    kernel_size=k,
                    stride=s,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input audios into patches.

        Args:
            x (torch.Tensor):
                Input audios. Shape is [batch_size, n_channels, sample_size].

        Returns:
            torch.Tensor:
                Output patches. Shape is [batch_size, n_patches, embed_dim].
        """
        x = self.conv_layers(x)
        # [batch_size, embed_dim, n_patches] -> [batch_size, n_patches, embed_dim]
        x = x.transpose(-1, -2)
        return x
