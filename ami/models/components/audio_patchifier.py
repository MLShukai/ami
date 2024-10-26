# Ref: https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py#L844

import dataclasses

import torch
import torch.nn as nn

@dataclasses.dataclass
class ConvLayerConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int

class TransposeLast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(-2, -1)
    
class AudioPatchifier(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, 
    ):
        super().__init__()
        
        # According to data2vec paper (https://arxiv.org/abs/2202.03555), 
        # 1D Convolutions with strides (5,2,2,2,2,2,2) and kernel widths (10,3,3,3,3,2,2)
        # results in window_size=400[samples] and hop_size=320[samples] as total.
        conv_layer_configs: list[ConvLayerConfig] = [
            ConvLayerConfig(in_channels=in_channels, out_channels=out_channels, kernel_size=10, stride=5),
            ConvLayerConfig(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2),
            ConvLayerConfig(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2),
            ConvLayerConfig(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2),
            ConvLayerConfig(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2),
            ConvLayerConfig(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2),
            ConvLayerConfig(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2),
        ]

        def block(
            n_in,
            n_out,
            k,
            stride,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=False)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            return nn.Sequential(
                make_conv(),
                nn.Sequential(
                    TransposeLast(),#x.transpose(-2, -1)
                    nn.LayerNorm(n_out, elementwise_affine=True),
                    TransposeLast(),#x.transpose(-2, -1)
                ),
                nn.GELU(),
            )

        self.conv_layers = nn.ModuleList()
        for conv_layer_config in conv_layer_configs:
            self.conv_layers.append(
                block(
                    conv_layer_config.in_channels,
                    conv_layer_config.out_channels,
                    conv_layer_config.kernel_size,
                    conv_layer_config.stride,
                )
            )

    def forward(self, x: torch.Tensor):
        """Convert input audios into patches.

        Args:
            x (torch.Tensor):
                Input audios. Shape is [batch_size, n_channels, n_samples].

        Returns:
            torch.Tensor:
                Output patches. Shape is [batch_size, n_patches, out_channels].
        """
        for conv in self.conv_layers:
            x = conv(x)
        # [batch_size, out_channels, n_patches]->[batch_size, n_patches, out_channels]
        x = x.transpose(-1, -2)
        return x