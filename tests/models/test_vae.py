from functools import partial

import pytest
import torch

from ami.models.components.small_conv_net import SmallConvNet
from ami.models.components.small_deconv_net import SmallDeconvNet
from ami.models.vae import VAE, Conv2dDecoder, Conv2dEncoder, EncoderWrapper

HEIGHT = 256
WIDTH = 256
CHANNELS = 3
DIM_EMBED = 256


class TestConv2dEncoder:
    def test_forward(self):
        mod = Conv2dEncoder(HEIGHT, WIDTH, CHANNELS, DIM_EMBED)
        x = torch.randn(8, CHANNELS, WIDTH, HEIGHT)
        z_dist = mod(x)
        assert isinstance(z_dist, torch.distributions.Normal)
        assert z_dist.rsample().shape == (8, DIM_EMBED)


class TestVAEDecoder:
    def test_forward(self):
        mod = Conv2dDecoder(HEIGHT, WIDTH, CHANNELS, DIM_EMBED)
        z = torch.randn(8, DIM_EMBED)
        rec_x = mod.forward(z)
        assert rec_x.shape == torch.Size((8, CHANNELS, HEIGHT, WIDTH))


class TestVAE:
    @pytest.fixture
    def encoder(self):
        encoder = Conv2dEncoder(HEIGHT, WIDTH, CHANNELS, DIM_EMBED)
        return encoder

    @pytest.fixture
    def decoder(self):
        decoder = Conv2dDecoder(HEIGHT, WIDTH, CHANNELS, DIM_EMBED)
        return decoder

    def test__init__(self, encoder, decoder):
        mod = VAE(encoder, decoder)
        assert mod.encoder is encoder
        assert mod.decoder is decoder

    def test__forward(self, encoder, decoder):
        mod = VAE(encoder, decoder)
        x = torch.randn(8, CHANNELS, HEIGHT, WIDTH)
        rec_x, z = mod(x)
        assert z.sample().shape == torch.Size((8, DIM_EMBED))


class TestEncoderWrapper:
    @pytest.fixture
    def encoder(self):
        encoder = Conv2dEncoder(HEIGHT, WIDTH, CHANNELS, DIM_EMBED)
        return encoder

    @pytest.fixture
    def encoder_wrapper(self, encoder):
        encoder_wrapper = EncoderWrapper(encoder)
        return encoder_wrapper

    def test__inference(self, encoder_wrapper):
        inference = encoder_wrapper.create_inference()
        x = torch.randn(16, CHANNELS, HEIGHT, WIDTH)
        z = inference(x)
        assert z.size() == (16, DIM_EMBED)
