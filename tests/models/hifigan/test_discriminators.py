import math

import pytest
import torch

from ami.models.hifigan.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)


class TestHifiGANDiscriminators:
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("channels", [1, 2])
    @pytest.mark.parametrize("initial_hidden_channels", [4])
    @pytest.mark.parametrize("sample_size", [8192])
    def test_multi_period_discriminator(
        self,
        batch_size: int,
        channels: int,
        initial_hidden_channels: int,
        sample_size: int,
    ):
        hifigan_discriminator = MultiPeriodDiscriminator(
            in_channels=channels, initial_hidden_channels=initial_hidden_channels
        )
        # define input wav
        input_waveforms = torch.randn([batch_size, channels, sample_size])
        # discriminate authenticity of the input audio
        authenticity_list, fmaps_list = hifigan_discriminator(input_waveforms)
        # check size of output authenticities
        for authenticity in authenticity_list:
            assert authenticity.dim() == 2
            assert authenticity.size(0) == batch_size
        # check size of output feature maps
        for fmaps in fmaps_list:
            for fmap in fmaps:
                assert fmap.dim() == 4
                assert fmap.size(0) == batch_size

    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("channels", [1, 2])
    @pytest.mark.parametrize("initial_hidden_channels", [16])
    @pytest.mark.parametrize("n_scales", [3])
    @pytest.mark.parametrize("sample_size", [8192])
    def test_multi_scale_discriminator(
        self,
        batch_size: int,
        channels: int,
        initial_hidden_channels: int,
        n_scales: int,
        sample_size: int,
    ):
        assert initial_hidden_channels % 16 == 0
        hifigan_discriminator = MultiScaleDiscriminator(
            in_channels=channels,
            initial_hidden_channels=initial_hidden_channels,
            n_scales=n_scales,
        )
        # define input wav
        input_waveforms = torch.randn([batch_size, channels, sample_size])
        # discriminate authenticity of the input audio
        authenticity_list, fmaps_list = hifigan_discriminator(input_waveforms)
        # check size of output authenticities
        for authenticity in authenticity_list:
            assert authenticity.dim() == 2
            assert authenticity.size(0) == batch_size
        # check size of output feature maps
        for fmaps in fmaps_list:
            for fmap in fmaps:
                assert fmap.dim() == 3
                assert fmap.size(0) == batch_size
