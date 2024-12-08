import math

import pytest
import torch

from ami.models.hifigan_discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator


class TestHifiGANDiscriminators:
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("channels", [1, 2])
    @pytest.mark.parametrize("sample_size", [8192])
    def test_multi_period_discriminator(
        self,
        batch_size: int,
        channels: int,
        sample_size: int,
    ):
        hifigan_discriminator = MultiPeriodDiscriminator(in_channels=channels)
        # define input wav
        real_waveforms = torch.randn([batch_size, channels, sample_size])
        fake_waveforms = torch.randn([batch_size, channels, sample_size])
        # generate waveforms
        y_d_rs, y_d_fs, fmaps_rs, fmaps_fs = hifigan_discriminator(real_waveforms, fake_waveforms)
        # check size of output discriminations
        for y_d_r, y_d_f in zip(y_d_rs, y_d_fs):
            assert y_d_r.dim()==2
            assert y_d_r.size(0)==batch_size
            assert y_d_r.size()==y_d_f.size()
        # check size of output feature maps
        for fmaps_r, fmaps_f in zip(fmaps_rs, fmaps_fs):
            for fmap_r, fmap_f in zip(fmaps_r, fmaps_f):
                assert fmap_r.dim()==4
                assert fmap_r.size(0)==batch_size
                assert fmap_r.size()==fmap_f.size()
        
        

