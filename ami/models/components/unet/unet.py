# Ref: https://github.com/facebookresearch/RCDM

import math

import torch
import torch.nn as nn

from .decoder_block import UnetDecoderBlock
from .encoder_block import UnetEncoderBlock
from .middle_block import UnetMiddleBlock


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps (torch.Tensor):
            Timesteps for noise reduction.
            (shape: [batch_size])
        dims (int):
            Output dims.
        max_period (int):
            Controls the minimum frequency of the embeddings.
    Returns:
        torch.Tensor:
            Embedded timesteps.
            (shape: [batch_size, dim])
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class UNet(nn.Module):
    """Unet architecture for noise prediction from input images."""

    def __init__(
        self,
        in_channels: int,
        timestep_hidden_dim: int,
        timestep_embed_dim: int,
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
            timestep_hidden_dim (int):
                Hidden dims of timestep embedding.
            timestep_embed_dim (int):
                Dims of timestep embedding.
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

        # embedding for timestep
        self.timestep_hidden_dim = timestep_hidden_dim
        self.timestep_emb = nn.Sequential(
            nn.Linear(timestep_hidden_dim, timestep_embed_dim),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, timestep_embed_dim),
        )
        # embedding for ssl latents
        self.ssl_emb = nn.Linear(ssl_latent_dim, timestep_embed_dim)

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
                    timestep_embed_dim=timestep_embed_dim,
                    out_channels=blocks_out_channels,
                    n_res_blocks=n_res_blocks,
                    use_attention=(layer_depth >= attention_start_depth),
                    num_heads=num_heads,
                    use_downsample=(not is_last_layer),
                )
            )
            in_channels_for_skip_connections += [blocks_out_channels for _ in range(n_res_blocks + (not is_last_layer))]

        # create middle blocks
        middle_block_channels = encoder_blocks_in_and_out_channels[-1][-1]
        self.middle_block = UnetMiddleBlock(
            in_and_out_channels=middle_block_channels,
            timestep_embed_dim=timestep_embed_dim,
            num_heads=num_heads,
        )

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
            self.decoder_blocks.append(
                UnetDecoderBlock(
                    in_channels=blocks_in_channels,
                    in_channels_for_skip_connections=in_channels_for_skip_connections[
                        -self.n_decoder_block_resblocks :
                    ][::-1],
                    timestep_embed_dim=timestep_embed_dim,
                    out_channels=blocks_out_channels,
                    n_res_blocks=self.n_decoder_block_resblocks,
                    use_attention=(layer_depth >= attention_start_depth),
                    num_heads=num_heads,
                    use_upsample=(not layer_depth == 0),
                )
            )
            in_channels_for_skip_connections = in_channels_for_skip_connections[: -self.n_decoder_block_resblocks]

        assert input_ch == decoder_blocks_in_and_out_channels[-1][-1]
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
        input_images: torch.Tensor,
        timesteps: torch.Tensor,
        latents_from_ssl: torch.Tensor,
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
