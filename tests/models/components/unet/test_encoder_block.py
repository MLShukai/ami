import pytest
import torch

from ami.models.components.unet.encoder_block import UnetEncoderBlock


class TestUnetEncoderBlock:
    # model params
    @pytest.mark.parametrize("in_channels", [32, 64])
    @pytest.mark.parametrize("timestep_embed_dim", [32])
    @pytest.mark.parametrize("out_channels", [32, 64])
    @pytest.mark.parametrize("n_res_blocks", [2])
    @pytest.mark.parametrize("use_attention", [False, True])
    @pytest.mark.parametrize("use_downsample", [False, True])
    # test input params
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.parametrize("height", [4, 32])
    @pytest.mark.parametrize("width", [4, 32])
    def test_unet_encoder_block(
        self,
        in_channels: int,
        timestep_embed_dim: int,
        out_channels: int,
        n_res_blocks: int,
        use_attention: bool,
        use_downsample: bool,
        batch_size: int,
        height: int,
        width: int,
    ):
        assert height % 2 == 0 and width % 2 == 0
        # define model
        unet_encoder_block = UnetEncoderBlock(
            in_channels=in_channels,
            timestep_embed_dim=timestep_embed_dim,
            out_channels=out_channels,
            n_res_blocks=n_res_blocks,
            use_attention=use_attention,
            num_heads=4,
            use_downsample=use_downsample,
        )
        # prepare inputs
        input_features = torch.randn([batch_size, in_channels, height, width])
        timestep_emb = torch.randn([batch_size, timestep_embed_dim])
        # get output
        output_features, output_features_for_skip_connections = unet_encoder_block(
            x=input_features, timestep_emb=timestep_emb
        )

        # check about output_features
        expected_height, expected_width = (height // 2, width // 2) if use_downsample else (height, width)
        assert output_features.size() == (batch_size, out_channels, expected_height, expected_width)

        # check about output_features_for_skip_connections
        assert isinstance(output_features_for_skip_connections, list)
        # check num of elements
        expected_n_features = n_res_blocks + int(use_downsample)
        assert len(output_features_for_skip_connections) == expected_n_features
        # check elements from resblocks
        for i in range(n_res_blocks):
            features = output_features_for_skip_connections[i]
            isinstance(features, torch.Tensor)
            assert features.size() == (batch_size, out_channels, height, width)
        # check element from downsample
        if use_downsample:
            features = output_features_for_skip_connections[-1]
            isinstance(features, torch.Tensor)
            assert features.size() == (batch_size, out_channels, height // 2, width // 2)
