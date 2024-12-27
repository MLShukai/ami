import math

import pytest
import torch

from ami.models.hifigan.hifigan_generator import HifiGANGenerator


class TestHifiGANGenerator:
    # model params
    @pytest.mark.parametrize(
        [
            "in_channels",
            "out_channels",
            "upsample_rates",
            "upsample_kernel_sizes",
            "upsample_paddings",
            "upsample_initial_channel",
            "resblock_kernel_sizes",
            "resblock_dilation_sizes",
        ],
        [
            [
                512,
                1,
                [8, 8, 2, 2],
                [16, 16, 4, 4],
                [
                    4,
                    4,
                    1,
                    1,
                ],  # Parameters not present in the original implementation. Calculated with (upsample_kernel_size - upsample_rate) // 2
                512,
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            ],  # cited from https://github.com/jik876/hifi-gan/blob/master/config_v1.json
            [
                512,
                2,
                [10, 8, 2, 2],
                [20, 16, 4, 4],
                [4, 2, 1, 1],
                512,
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            ],  # Equivalent to window_size=400 and hop_size=320.
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
        upsample_paddings: list[int],
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
            upsample_paddings=upsample_paddings,
            upsample_initial_channel=upsample_initial_channel,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
        )
        # define features
        input_features = torch.randn([batch_size, in_channels, frames])
        # generate waveforms
        output_waveforms = hifigan_generator(features=input_features)
        # check size of output waveforms
        assert output_waveforms.size(0) == batch_size, "batch_size mismatch"
        assert output_waveforms.size(1) == out_channels, "out_channels of output waveform mismatch"
        hop_size = math.prod(upsample_rates)
        window_size = _calc_window_size(
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            upsample_paddings=upsample_paddings,
        )
        expected_sample_size = (frames - 1) * hop_size + window_size
        assert output_waveforms.size(2) == expected_sample_size, "sample size of output waveform mismatch"


def _calc_window_size(
    upsample_rates: list[int],
    upsample_kernel_sizes: list[int],
    upsample_paddings: list[int],
) -> int:
    """Calculate window_size successive deconv layers correspond to as a whole.
    Ref: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html

    Args:
        upsample_rates (list[int]): Strides of deconvs.
        upsample_kernel_sizes (list[int]): Kernel sizes of deconvs.
        upsample_paddings (list[int]): Padding sizes of deconvs.

    Returns:
        int: Window size as total.
    """
    assert len(upsample_rates) == len(upsample_kernel_sizes) == len(upsample_paddings)
    n_deconvs = len(upsample_rates)
    window_size = 1
    for i in range(n_deconvs):
        window_size += (upsample_kernel_sizes[i] - 1 - 2 * upsample_paddings[i]) * math.prod(upsample_rates[i + 1 :])
    return window_size
