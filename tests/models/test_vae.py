from functools import partial

import pytest
import torch

from ami.models.components.small_conv_net import SmallConvNet
from ami.models.components.small_deconv_net import SmallDeconvNet
from ami.models.vae import VAE, Decoder, Encoder, EncoderWrapper

HEIGHT = 256
WIDTH = 256
CHANNELS = 3
DIM_EMBED = 256


class TestVAEEncoder:
    @pytest.fixture
    def small_conv_net(self):
        net = SmallConvNet(HEIGHT, WIDTH, CHANNELS, 2 * DIM_EMBED)
        return net

    def test__init__(self, small_conv_net):
        mod = Encoder(small_conv_net)
        assert mod.conv_net is small_conv_net

    def test_forward(self, small_conv_net):
        mod = Encoder(small_conv_net)
        x = torch.randn(8, CHANNELS, WIDTH, HEIGHT)
        z_dist = mod(x)
        assert isinstance(z_dist, torch.distributions.Normal)
        assert z_dist.rsample().shape == (8, DIM_EMBED)


class TestVAEDecoder:
    @pytest.fixture
    def small_deconv_net(self):
        net = SmallDeconvNet(HEIGHT, WIDTH, CHANNELS, DIM_EMBED)
        return net

    def test__init__(self, small_deconv_net):
        mod = Decoder(small_deconv_net)
        assert mod.deconv_net is small_deconv_net

    def test_forward(self, small_deconv_net):
        mod = Decoder(small_deconv_net)
        z = torch.randn(8, DIM_EMBED)
        rec_x = mod.forward(z)
        assert rec_x.shape == torch.Size((8, CHANNELS, HEIGHT, WIDTH))


class TestVAE:
    @pytest.fixture
    def encoder(self):
        encoder = Encoder(SmallConvNet(HEIGHT, WIDTH, CHANNELS, 2 * DIM_EMBED))
        return encoder

    @pytest.fixture
    def decoder(self):
        decoder = Decoder(SmallDeconvNet(HEIGHT, WIDTH, CHANNELS, DIM_EMBED))
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
        encoder = Encoder(SmallConvNet(HEIGHT, WIDTH, CHANNELS, 2 * DIM_EMBED))
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
