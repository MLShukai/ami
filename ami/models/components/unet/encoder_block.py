# Ref: https://github.com/facebookresearch/RCDM

import torch
import torch.nn as nn

from .commons import AttentionBlock, ResBlock


class UnetEncoderBlock(nn.Module):
    """A block for Unet encoder."""

    def __init__(
        self,
        in_channels: int,
        timestep_embed_dim: int,
        out_channels: int,
        n_res_blocks: int,
        use_attention: bool,
        num_heads: int,
        use_downsample: bool,
    ) -> None:
        """A block for Unet encoder.

        Args:
            in_channels (int):
                Input num of channels.
            timestep_embed_dim (int):
                Dims of timestep embedding.
            out_channels (int):
                Output num of channels.
            n_res_blocks (int):
                Num of resblocks in this block.
            use_attention (bool):
                Whether or not an attention is added after each resblock
            num_heads (int):
                num_head when adding attention layers.
            use_downsample (bool):
                Whether downsample at the end in self.forward.
        """
        super().__init__()

        self.n_res_blocks = n_res_blocks
        self.use_attention = use_attention
        # define resblocks
        self.resnet_layers = nn.ModuleList([])
        self.attention_layers = nn.ModuleList([])
        for res_block_i in range(n_res_blocks):
            self.resnet_layers.append(
                ResBlock(
                    in_channels=(in_channels if res_block_i == 0 else out_channels),
                    emb_channels=timestep_embed_dim,
                    out_channels=out_channels,
                )
            )
            if use_attention:
                self.attention_layers.append(AttentionBlock(out_channels, num_heads=num_heads))

        # prepare downsample layer if needed
        self.conv_for_downsample = (
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1) if use_downsample else None
        )

    def forward(self, x: torch.Tensor, timestep_emb: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """apply this Encoder block.

        Args:
            x (torch.Tensor):
                Input features.
                (shape: [batch_size, in_channels, height, width])
            timestep_emb (torch.Tensor):
                Embedded timesteps.
                (shape: [batch_size, timestep_embed_dim])
        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]
                1. Output features.
                    If use_downsample is True, shape is
                    [batch_size, out_channels, height//2, width//2],
                    otherwise [batch_size, out_channels, height, width].
                2. Features for skip connections in Decoder blocks.
                    Each shape: [batch_size, out_channels, height, width]).
                    If use_downsample is True, shape of the last element is
                    [batch_size, out_channels, height//2, width//2].
        """
        features_for_skip_connections: list[torch.Tensor] = []
        # apply resblocks
        feature = x
        for i in range(self.n_res_blocks):
            feature = self.resnet_layers[i](feature, timestep_emb)
            if self.use_attention:
                feature = self.attention_layers[i](feature)
            # keep features for skip connection to Decoder
            features_for_skip_connections.append(feature)
        # downsamples features if needed
        if self.conv_for_downsample is not None:
            feature = self.conv_for_downsample(feature)
            # keep features for skip connection to Decoder
            features_for_skip_connections.append(feature)
        return feature, features_for_skip_connections
