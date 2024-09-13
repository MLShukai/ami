import math

import torch
import torch.nn as nn

from .utils import size_2d, size_2d_to_int_tuple


class ResBlock(nn.Module):
    """A residual block for IJEPALatentVisualizationDecoder Components."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_norm_groups_in_input_layer: int = 32,
        num_norm_groups_in_output_layer: int = 32,
    ) -> None:
        """A residual block for IJEPALatentVisualizationDecoder Components.

        Args:
            in_channels (int):
                Input num of channels.
            out_channels (int):
                Output num of channels.
            num_norm_groups_in_input_layer (int):
                num of groups in nn.GroupNorm in self.in_layers.
                Default to 32.
            num_norm_groups_in_output_layer (int):
                num of groups in nn.GroupNorm in self.out_layers.
                Default to 32.
        """
        super().__init__()
        assert in_channels % num_norm_groups_in_input_layer == 0
        assert out_channels % num_norm_groups_in_output_layer == 0
        # input layers
        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=num_norm_groups_in_input_layer, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        # output layers
        self.out_group_norm = nn.GroupNorm(num_groups=num_norm_groups_in_output_layer, num_channels=out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        # initialize conv layer as zeros
        for p in self.out_layers[-1].parameters():
            p.detach().zero_()

        self.skip_connection: nn.Module = (
            nn.Identity() if out_channels == in_channels else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to input Tensor.

        Args:
            x (torch.Tensor):
                Input features.
                (shape: [batch_size, in_channels, height, width])
        Returns:
            torch.Tensor:
                Output features.
                (shape: [batch_size, out_channels, height, width])
        """
        h = self.in_layers(x)
        h = self.out_group_norm(h)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class DecoderBlock(nn.Module):
    """A block for decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_res_blocks: int,
        use_upsample: bool,
    ) -> None:
        """A block for IJEPALatentVisualizationDecoderr.

        Args:
            in_channels (int):
                Input num of channels.
            out_channels (int):
                Output num of channels.
            n_res_blocks (int):
                Num of resblocks in this block.
            use_upsample (bool):
                Whether upsample at the end in self.forward.
        """
        super().__init__()

        self.n_res_blocks = n_res_blocks
        # define resblocks
        self.resnet_layers = nn.ModuleList([])
        self.attention_layers = nn.ModuleList([])
        for res_block_i in range(n_res_blocks):
            self.resnet_layers.append(
                ResBlock(
                    in_channels=(in_channels if res_block_i == 0 else out_channels),
                    out_channels=out_channels,
                )
            )

        # prepare upsample layer if needed
        self.conv_for_upsample = nn.Conv2d(out_channels, out_channels, 3, padding=1) if use_upsample else None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """apply this Decoder block.

        Args:
            x (torch.Tensor):
                Input features.
                (shape: [batch_size, in_channels, height, width])
        Returns:
            torch.Tensor
                Output features.
                    If use_upsample is True, shape is
                    [batch_size, out_channels, height*2, width*2],
                    otherwise [batch_size, out_channels, height, width].
        """
        # apply resblocks
        feature = x
        for i in range(self.n_res_blocks):
            feature = self.resnet_layers[i](feature)
        # upsamples features if needed
        if self.conv_for_upsample is not None:
            feature = nn.functional.interpolate(feature, scale_factor=2, mode="nearest")
            feature = self.conv_for_upsample(feature)
        return feature


class IJEPALatentVisualizationDecoder(nn.Module):
    """Decoder for visualizing latents as image."""

    def __init__(
        self,
        input_n_patches: size_2d,
        input_latents_dim: int,
        decoder_blocks_in_and_out_channels: list[tuple[int, int]],
        n_res_blocks: int,
        num_heads: int,
    ) -> None:
        """Decoder for visualizing latents as image.

        Args:
            input_n_patches (size_2d):
                Input latents num of patches (height, width).
            input_latents_dim (int):
                Input latents dim.
            decoder_blocks_in_and_out_channels (list[tuple[int, int]]):
                Specify (in_channels, out_channels) of each decoder block,
                in order from input block to output block.
            n_res_blocks (int):
                Num of resblocks in each decoder block.
            num_heads (int):
                num_head when adding attention layers.
        """
        super().__init__()

        self.input_n_patches = size_2d_to_int_tuple(input_n_patches)
        self.input_latents_dim = input_latents_dim

        # define input blocks
        decoder_blocks_first_n_channels = decoder_blocks_in_and_out_channels[0][0]
        self.input_resblock_1 = ResBlock(
            in_channels=input_latents_dim,
            out_channels=decoder_blocks_first_n_channels,
        )
        self.input_attention = nn.MultiheadAttention(
            embed_dim=decoder_blocks_first_n_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.input_resblock_2 = ResBlock(
            in_channels=decoder_blocks_first_n_channels,
            out_channels=decoder_blocks_first_n_channels,
        )

        # define decoder blocks
        self.decoder_blocks = nn.Sequential(
            *[
                DecoderBlock(
                    in_channels=blocks_in_channels,
                    out_channels=blocks_out_channels,
                    n_res_blocks=n_res_blocks,
                    use_upsample=(not (i == (len(decoder_blocks_in_and_out_channels) - 1))),
                )
                for i, (blocks_in_channels, blocks_out_channels) in enumerate(decoder_blocks_in_and_out_channels)
            ]
        )

        # define output blocks
        decoder_blocks_last_n_channels = decoder_blocks_in_and_out_channels[-1][-1]
        self.output_layer = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=decoder_blocks_last_n_channels),
            nn.SiLU(),
            nn.Conv2d(decoder_blocks_last_n_channels, 3, kernel_size=3, padding=1),
        )
        # initialize conv layer as zeros
        for p in self.output_layer[-1].parameters():
            p.detach().zero_()

    def forward(
        self,
        input_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Generate images from input latents.

        Args:
            input_latents (torch.Tensor):
                latents from other decoder model.
                (shape: [batch_size, n_patches_height * n_patches_width, latents_dim])
        Returns:
            torch.Tensor:
                Generated images.
                (shape:
                    [
                        batch_size,
                        3,
                        n_patches_height * (2**(len(decoder_blocks_in_and_out_channels)-1)),
                        n_patches_width * (2**(len(decoder_blocks_in_and_out_channels)-1)),
                    ]
                )
        """

        # reshape input latents
        batch_size = input_latents.size(0)
        height, width = self.input_n_patches
        input_latents = torch.reshape(input_latents, (batch_size, height, width, self.input_latents_dim))
        input_latents = input_latents.movedim(-1, 1)  # [batch, latent, height, width]
        # apply input layers
        feature = self.input_resblock_1(input_latents)
        batch_size, channels, height, width = feature.size()
        feature = torch.reshape(feature.movedim(1, -1), (batch_size, height * width, channels))
        feature, _ = self.input_attention(query=feature, key=feature, value=feature)
        feature = torch.reshape(feature.movedim(-1, 1), (batch_size, channels, height, width))
        feature = self.input_resblock_2(feature)
        # apply decoder layers
        feature = self.decoder_blocks(feature)
        # apply output layers
        output = self.output_layer(feature)
        return output
