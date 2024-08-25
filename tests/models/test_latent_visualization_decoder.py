import random
from typing import Optional

import pytest
import torch

from ami.models.latent_visualization_decoder import (
    ResBlock, DecoderBlock, LatentVisualizationDecoder
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
    @pytest.mark.parametrize("n_res_blocks", [3])
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
            use_upsample=use_upsample,
        )
        # define sample input latents
        inputs = torch.randn([batch_size, in_channels, height, width])
        # get output latents
        outputs = resblock(x=inputs)
        # check size of output latents
        expected_height, expected_width = (height * 2, width * 2) if use_upsample else (height, width)
        assert outputs.size() == (batch_size, out_channels, expected_height, expected_width)
    
    # model params
    @pytest.mark.parametrize("input_n_patches_height", [1, 12])
    @pytest.mark.parametrize("input_n_patches_width", [1, 12])
    @pytest.mark.parametrize("input_latents_dim", [384])
    @pytest.mark.parametrize(
        "decoder_blocks_in_and_out_channels",
        [
            [(128, 128), (128, 96), (96, 64), (64, 32)],
            [(96, 96), (96, 64), (64, 32)],
        ],
    )
    @pytest.mark.parametrize("n_res_blocks", [3])
    @pytest.mark.parametrize("num_heads", [2])
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_latent_visualization_decoder(
        self,
        input_n_patches_height: int,
        input_n_patches_width: int,
        input_latents_dim: int,
        decoder_blocks_in_and_out_channels: list[tuple[int, int]],
        n_res_blocks: int,
        num_heads: int,
        batch_size: int,
    ):
        # define resblock
        latent_visualization_decoder = LatentVisualizationDecoder(
            input_n_patches = (input_n_patches_height, input_n_patches_width),
            input_latents_dim = input_latents_dim,
            decoder_blocks_in_and_out_channels=decoder_blocks_in_and_out_channels,
            n_res_blocks=n_res_blocks,
            num_heads=num_heads,
        )
        # define sample inputs
        n_patches = input_n_patches_height*input_n_patches_width
        latents = torch.randn([batch_size, n_patches, input_latents_dim])
        # get output images
        out_images = latent_visualization_decoder(latents)
        # check size of output latents
        magnification = 2**(len(decoder_blocks_in_and_out_channels)-1)
        expected_height = input_n_patches_height*magnification
        expected_width = input_n_patches_width*magnification
        assert out_images.size() == (batch_size, 3, expected_height, expected_width)

