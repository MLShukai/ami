# Ref: https://github.com/facebookresearch/ijepa

import math

import torch
import torch.nn as nn

from .components.unet.commons import AttentionBlock

class ResBlock(nn.Module):
    """A residual block for Unet Components."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_norm_groups_in_input_layer: int = 32,
        num_norm_groups_in_output_layer: int = 32,
    ) -> None:
        """A residual block for Unet Components.

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
        """Apply the block to input Tensor, conditioned on emb.

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
        use_attention: bool,
        num_heads: int,
        use_upsample: bool,
    ) -> None:
        """A block for Unet decoder.

        Args:
            in_channels (int):
                Input num of channels.
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

        self.n_res_blocks = n_res_blocks
        self.use_attention = use_attention
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
            if use_attention:
                self.attention_layers.append(AttentionBlock(out_channels, num_heads=num_heads))

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
            if self.use_attention:
                feature = self.attention_layers[i](feature)
        # upsamples features if needed
        if self.conv_for_upsample is not None:
            feature = nn.functional.interpolate(feature, scale_factor=2, mode="nearest")
            feature = self.conv_for_upsample(feature)
        return feature


class LatentVisualizationDecoder(nn.Module):
    """Unet architecture for noise prediction from input images."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder_blocks_in_and_out_channels: list[tuple[int, int]],
        n_res_blocks: int,
        attention_start_depth: int,
        num_heads: int,
        ssl_latent_dim: int,
    ) -> None:
        """Unet architecture for noise prediction from input images.

        Args:
            in_channels (int):
                Input images num of channels.
            out_channels (int):
                Output images num of channels.
            encoder_blocks_in_and_out_channels (list[tuple[int, int]]):
                Specify (in_channels, out_channels) of each encoder block,
                in order from the shallowest block to the deepest block.
            n_res_blocks (int):
                num of resblocks in each encoder block.
            attention_start_depth (int):
                Append attention layers for blocks deeper than attention_start_depth. 0 origin.
            num_heads (int):
                num_head when adding attention layers.
            ssl_latent_dim (int):
                Dims of latents from ssl model.
        """
        super().__init__()

        # create encoder blocks
        self.n_res_blocks = n_res_blocks
        input_ch = encoder_blocks_in_and_out_channels[0][0]
        self.input_layer = nn.Conv2d(in_channels, input_ch, 3, padding=1)
        in_channels_for_skip_connections: list[int] = [input_ch]
        self.encoder_blocks = nn.ModuleList([])
        for layer_depth, (blocks_in_channels, blocks_out_channels) in enumerate(encoder_blocks_in_and_out_channels):
            is_last_layer = layer_depth == len(encoder_blocks_in_and_out_channels) - 1
            self.encoder_blocks.append(
                UnetEncoderBlock(
                    in_channels=blocks_in_channels,
                    out_channels=blocks_out_channels,
                    n_res_blocks=n_res_blocks,
                    use_attention=(layer_depth >= attention_start_depth),
                    num_heads=num_heads,
                    use_downsample=(not is_last_layer),
                )
            )
            in_channels_for_skip_connections += [blocks_out_channels for _ in range(n_res_blocks + (not is_last_layer))]

        # create decoder blocks in contrast to encoder blocks
        decoder_blocks_in_and_out_channels: list[tuple[int, int]] = []
        # channels of first UnetDecoderBlock are same as middle block
        decoder_blocks_in_and_out_channels.append((middle_block_channels, middle_block_channels))
        # check channels from last to second encoder block
        for (encoder_block_in_channel, encoder_block_out_channel) in encoder_blocks_in_and_out_channels[::-1][:-1]:
            # reverse in and out channels between encoder and decoder
            decoder_block_in_channel = encoder_block_out_channel
            decoder_block_out_channel = encoder_block_in_channel
            decoder_blocks_in_and_out_channels.append((decoder_block_in_channel, decoder_block_out_channel))

        self.n_decoder_block_resblocks = n_res_blocks + 1
        self.decoder_blocks = nn.ModuleList([])
        for layer_depth, (blocks_in_channels, blocks_out_channels) in zip(
            reversed(range(len(decoder_blocks_in_and_out_channels))), decoder_blocks_in_and_out_channels
        ):
            # layer_depth is [len(decoder_blocks_in_and_out_channels), len(decoder_blocks_in_and_out_channels) - 1, ..., 0]
            is_last_layer = layer_depth == 0
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=blocks_in_channels,
                    out_channels=blocks_out_channels,
                    n_res_blocks=self.n_decoder_block_resblocks,
                    use_attention=(layer_depth >= attention_start_depth),
                    num_heads=num_heads,
                    use_upsample=(not is_last_layer),
                )
            )

        self.output_layer = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=input_ch),
            nn.SiLU(),
            nn.Conv2d(input_ch, out_channels, 3, padding=1),
        )
        # initialize conv layer as zeros
        for p in self.output_layer[-1].parameters():
            p.detach().zero_()

    def forward(
        self,
        input_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise components from input images, depending on timesteps,
        conditioned with latents_from_ssl.

        Args:
            input_images (torch.Tensor):
                Input noisy images.
                (shape: [batch_size, in_channels, height, width])
            timesteps (torch.Tensor):
                Timesteps for noise reduction.
                (shape: [batch_size])
            latents_from_ssl (torch.Tensor):
                Latents from another ssl model.
                (shape: [batch_size, n_patches, ssl_latent_dim])
        Returns:
            torch.Tensor:
                Predicted noise.
                (shape: [batch_size, out_channels, height, width])
        """

        # Embed timestep
        emb = self.timestep_emb(timestep_embedding(timesteps, self.timestep_hidden_dim))
        # Embed latents from ssl model.
        latents_from_ssl = self.ssl_emb(latents_from_ssl)
        latents_from_ssl = latents_from_ssl.mean(dim=1)
        emb = emb + latents_from_ssl

        features_for_skip_connections: list[torch.Tensor] = []
        # Input Conv
        feature = self.input_layer(input_images)
        features_for_skip_connections.append(feature)
        # Unet Encoder
        for encoder_block in self.encoder_blocks:
            feature, _features_for_skip_connections = encoder_block(feature, emb)
            features_for_skip_connections += _features_for_skip_connections
        # Unet Middle Block
        feature = self.middle_block(feature, emb)
        # Unet Decoder
        features_for_skip_connections = features_for_skip_connections[::-1]
        for decoder_block in self.decoder_blocks:
            feature = decoder_block(feature, emb, features_for_skip_connections[: self.n_decoder_block_resblocks])
            features_for_skip_connections = features_for_skip_connections[self.n_decoder_block_resblocks :]
        # Output Conv
        output = self.output_layer(feature)
        return output
