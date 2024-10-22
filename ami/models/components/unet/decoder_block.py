# Ref: https://github.com/facebookresearch/RCDM

import torch
import torch.nn as nn

from .commons import AttentionBlock, ResBlock


class UnetDecoderBlock(nn.Module):
    """A block for Unet decoder."""

    def __init__(
        self,
        in_channels: int,
        in_channels_for_skip_connections: list[int],
        timestep_embed_dim: int,
        out_channels: int,
        n_res_blocks: int,
        use_attention: bool,
        num_heads: int,
        use_upsample: bool,
    ) -> None:
        """A block for Unet decoder.

        Args:
            in_channels (int):
                Input num of channels.
            in_channels_for_skip_connections (list[int]):
                Dims of each features for skip connections from encoder blocks.
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
            use_upsample (bool):
                Whether upsample at the end in self.forward.
        """
        super().__init__()
        assert len(in_channels_for_skip_connections) == n_res_blocks

        self.n_res_blocks = n_res_blocks
        self.use_attention = use_attention
        # define resblocks
        self.resnet_layers = nn.ModuleList([])
        self.attention_layers = nn.ModuleList([])
        for res_block_i in range(n_res_blocks):
            self.resnet_layers.append(
                ResBlock(
                    in_channels=(in_channels if res_block_i == 0 else out_channels)
                    + in_channels_for_skip_connections[res_block_i],
                    emb_channels=timestep_embed_dim,
                    out_channels=out_channels,
                )
            )
            if use_attention:
                self.attention_layers.append(AttentionBlock(out_channels, num_heads=num_heads))

        # prepare upsample layer if needed
        self.conv_for_upsample = nn.Conv2d(out_channels, out_channels, 3, padding=1) if use_upsample else None

    def forward(
        self,
        x: torch.Tensor,
        timestep_emb: torch.Tensor,
        features_for_skip_connections: list[torch.Tensor],
    ) -> torch.Tensor:
        """Apply this Decoder block.

        Args:
            x (torch.Tensor):
                Input features.
                (shape: [batch_size, in_channels, height, width])
            timestep_emb (torch.Tensor):
                Embedded timesteps.
                (shape: [batch_size, timestep_embed_dim])
            features_for_skip_connections (list[torch.Tensor])
                Features from encoder blocks.
                (each shape: [batch_size, in_channels_for_skip_connections[i], height, width])
        Returns:
            torch.Tensor
                Output features.
                    If use_upsample is True, shape is
                    [batch_size, out_channels, height*2, width*2],
                    otherwise [batch_size, out_channels, height, width].
        """
        assert self.n_res_blocks == len(features_for_skip_connections)
        # apply resblocks
        feature = x
        for i in range(self.n_res_blocks):
            feature = torch.cat([feature, features_for_skip_connections[i]], dim=1)
            feature = self.resnet_layers[i](feature, timestep_emb)
            if self.use_attention:
                feature = self.attention_layers[i](feature)
        # upsamples features if needed
        if self.conv_for_upsample is not None:
            feature = nn.functional.interpolate(feature, scale_factor=2, mode="nearest")
            feature = self.conv_for_upsample(feature)
        return feature
