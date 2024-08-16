# Ref: https://github.com/facebookresearch/RCDM


import torch
import torch.nn as nn

from .commons import AttentionBlock, ResBlock


class UnetMiddleBlock(nn.Module):
    """A block between encoder and decoder in Unet."""

    def __init__(
        self,
        in_and_out_channels: int,
        timestep_embed_dim: int,
        num_heads: int,
    ) -> None:
        """A block between encoder and decoder in Unet.

        Args:
            in_and_out_channels (int):
                Input and output num of channels.
            timestep_embed_dim (int):
                Dims of timestep embedding.
            num_heads (int):
                num_head when adding attention layers.
        """
        super().__init__()

        self.resblock_in = ResBlock(
                    in_channels=in_and_out_channels, emb_channels=timestep_embed_dim, out_channels=in_and_out_channels
                )
        self.resblock_out = ResBlock(
                    in_channels=in_and_out_channels, emb_channels=timestep_embed_dim, out_channels=in_and_out_channels
                )
        self.attention = AttentionBlock(in_and_out_channels, num_heads=num_heads)]

    def forward(self, x: torch.Tensor, timestep_emb: torch.Tensor) -> torch.Tensor:
        """apply this Middle block.

        Args:
            x (torch.Tensor):
                Input features.
                (shape: [batch_size, in_and_out_channels, height, width])
            timestep_emb (torch.Tensor):
                Embedded timesteps.
                (shape: [batch_size, timestep_embed_dim])
        Returns:
            torch.Tensor
                Output features.
                (shape: [batch_size, in_and_out_channels, height, width])
        """
        assert len(self.resnet_blocks) == len(self.attentions) + 1
    x = self.resblock_in(x, timestep_emb)
    x = self.attention(x)
    x = self.resblock_out(x, timestep_emb)
        return x
