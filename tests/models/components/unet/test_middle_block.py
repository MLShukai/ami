import pytest
import torch

from ami.models.components.unet.middle_block import UnetMiddleBlock


class TestUnetMiddleBlock:
    # model params
    @pytest.mark.parametrize("in_and_out_channels", [32, 64])
    @pytest.mark.parametrize("timestep_embed_dim", [32])
    # test input params
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.parametrize("height", [4, 32])
    @pytest.mark.parametrize("width", [4, 32])
    def test_unet_middle_block(
        self,
        in_and_out_channels: int,
        timestep_embed_dim: int,
        batch_size: int,
        height: int,
        width: int,
    ):
        assert height % 2 == 0 and width % 2 == 0
        # define model
        unet_middle_block = UnetMiddleBlock(
            in_and_out_channels=in_and_out_channels,
            timestep_embed_dim=timestep_embed_dim,
            num_heads=4,
        )
        # prepare inputs
        input_features = torch.randn([batch_size, in_and_out_channels, height, width])
        timestep_emb = torch.randn([batch_size, timestep_embed_dim])
        # get output
        output_features = unet_middle_block(
            x=input_features,
            timestep_emb=timestep_emb,
        )

        # check about output_features
        assert output_features.size() == (batch_size, in_and_out_channels, height, width)
