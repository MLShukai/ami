import pytest
import torch

from ami.models.components.small_conv_net import SmallConvNet
from ami.models.components.small_deconv_net import SmallDeconvNet
from ami.models.vae import VAE, Decoder, Encoder
from ami.models.vae_wrapper import VAEWrapper

channels = 3
height = 256
width = 256
dim = 128
small_conv_net = SmallConvNet(height, width, channels, dim * 2)
small_deconv_net = SmallDeconvNet(height, width, channels, dim)
encoder = Encoder(small_conv_net)
decoder = Decoder(small_deconv_net)
vae = VAE(encoder, decoder)


class TestVAEWrapper:
    def test_forward(self):
        wrapper = VAEWrapper(vae)
        x = torch.randn(16, channels, height, width)
        x_reconstructed, dist = wrapper(x)
        assert x.size() == x_reconstructed.size()

    def test_inference(self):
        wrapper = VAEWrapper(vae)
        inference = wrapper.create_inference()
        x = torch.randn(16, channels, height, width)
        sample = inference.infer(x)
        assert sample.size() == (16, dim)
