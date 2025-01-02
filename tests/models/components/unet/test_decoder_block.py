import random

import pytest
import torch

from ami.models.components.unet.decoder_block import UnetDecoderBlock


class TestUnetDecoderBlock:
    # model params
    @pytest.mark.parametrize("in_channels", [32])
    @pytest.mark.parametrize("in_channels_for_skip_connections", [[32, 64, 96]])
    @pytest.mark.parametrize("timestep_embed_dim", [32])
    @pytest.mark.parametrize("out_channels", [32])
    @pytest.mark.parametrize("use_attention", [False, True])
    @pytest.mark.parametrize("use_upsample", [False, True])
    # test input params
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.parametrize("height", [32])
    @pytest.mark.parametrize("width", [32])
    def test_unet_decoder_block(
        self,
        in_channels: int,
        in_channels_for_skip_connections: list[int],
        timestep_embed_dim: int,
        out_channels: int,
        use_attention: bool,
        use_upsample: bool,
        batch_size: int,
        height: int,
        width: int,
    ):
        assert height % 2 == 0 and width % 2 == 0
        n_res_blocks = len(in_channels_for_skip_connections)
        # define model
        unet_decoder_block = UnetDecoderBlock(
            in_channels=in_channels,
            in_channels_for_skip_connections=in_channels_for_skip_connections,
            timestep_embed_dim=timestep_embed_dim,
            out_channels=out_channels,
            n_res_blocks=n_res_blocks,
            use_attention=use_attention,
            num_heads=4,
            use_upsample=use_upsample,
        )
        # prepare inputs
        input_features = torch.randn([batch_size, in_channels, height, width])
        timestep_emb = torch.randn([batch_size, timestep_embed_dim])
        # prepare features assumed as output from encoder blocks
        features_for_skip_connections = [
            torch.randn([batch_size, ch, height, width]) for ch in in_channels_for_skip_connections
        ]
        # get output
        output_features = unet_decoder_block(
            x=input_features,
            timestep_emb=timestep_emb,
            features_for_skip_connections=features_for_skip_connections,
        )

        # check about output_features
        expected_height, expected_width = (height * 2, width * 2) if use_upsample else (height, width)
        assert output_features.size() == (batch_size, out_channels, expected_height, expected_width)
