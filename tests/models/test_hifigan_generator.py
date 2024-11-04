import pytest
import torch

from ami.models.hifigan_generator import HifiGANGenerator


class TestHifiGANGenerator:
    # model params
    @pytest.mark.parametrize(
        [
            "in_channels", 
            "out_channels",
            "upsample_rates",
            "upsample_kernel_sizes",
            "upsample_initial_channel",
            "resblock_kernel_sizes",
            "resblock_dilation_sizes"
        ],
        [
            [
                512,
                1,
                [8, 8, 2, 2],
                [16, 16, 4, 4],
                128,
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            ],
        ],
    )
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("frames", [1, 50])
    def test_forward(
        self,
        in_channels: int,
        out_channels: int,
        upsample_rates: list[int],
        upsample_kernel_sizes: list[int],
        upsample_initial_channel: int,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        batch_size: int,
        frames: int,
    ):
        hifigan_generator = HifiGANGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            upsample_initial_channel=upsample_initial_channel,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
        )
        # define features
        input_features = torch.randn([batch_size, in_channels, frames])
        # generate waveforms
        output_waveforms = hifigan_generator(features=input_features)
        # check size of output latent
        assert output_waveforms.size(0) == batch_size, "batch_size mismatch"
        assert output_waveforms.size(1) == out_channels, "out_channels of output waveform mismatch"

