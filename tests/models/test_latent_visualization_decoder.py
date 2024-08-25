import random
from typing import Optional

import pytest
import torch

from ami.models.latent_visualization_decoder import (
    ResBlock, DecoderBlock
)


class TestLatentVisualizationDecoder:
    # model params
    @pytest.mark.parametrize("in_channels", [32, 128])
    @pytest.mark.parametrize("out_channels", [32, 128])
    @pytest.mark.parametrize("num_norm_groups_in_input_layer", [32])
    @pytest.mark.parametrize("num_norm_groups_in_output_layer", [32])
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("height", [4, 64])
    @pytest.mark.parametrize("width", [4, 64])
    def test_resblock(
        self,
        in_channels: int,
        out_channels: int,
        num_norm_groups_in_input_layer: int,
        num_norm_groups_in_output_layer: int,
        batch_size: int,
        height: int,
        width: int,
    ):
        # define resblock
        resblock = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            num_norm_groups_in_input_layer=num_norm_groups_in_input_layer,
            num_norm_groups_in_output_layer=num_norm_groups_in_output_layer,
        )
        # define sample inputs
        inputs = torch.randn([batch_size, in_channels, height, width])
        # get latents
        outputs = resblock(x=inputs)
        # check size of output latent
        assert outputs.shape == (batch_size, out_channels, height, width)
    
    # model params
    @pytest.mark.parametrize("in_channels", [32, 128])
    @pytest.mark.parametrize("out_channels", [32, 128])
    @pytest.mark.parametrize("n_res_blocks", [2])
    @pytest.mark.parametrize("use_attention", [True, False])
    @pytest.mark.parametrize("num_heads", [2])
    @pytest.mark.parametrize("use_upsample", [True, False])
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("height", [4, 64])
    @pytest.mark.parametrize("width", [4, 64])
    def test_decoder_block(
        self,
        in_channels: int,
        out_channels: int,
        n_res_blocks: int,
        use_attention: bool,
        num_heads: int,
        use_upsample: bool,
        batch_size: int,
        height: int,
        width: int,
    ):
        # define resblock
        resblock = DecoderBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            n_res_blocks=n_res_blocks,
            use_attention=use_attention,
            num_heads=num_heads,
            use_upsample=use_upsample,
        )
        # define sample input latents
        inputs = torch.randn([batch_size, in_channels, height, width])
        # get output latents
        outputs = resblock(x=inputs)
        # check size of output latents
        expected_height, expected_width = (height * 2, width * 2) if use_upsample else (height, width)
        assert outputs.size() == (batch_size, out_channels, expected_height, expected_width)
