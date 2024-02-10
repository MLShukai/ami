import pytest
import torch

from ami.models.components.small_deconv_net import SmallDeconvNet as cls


class TestSmallDeconvNet:
    def test__init__(self):
        mod = cls(256, 256, 3, 256, True, torch.nn.ReLU())
        assert mod.channels == 3
        assert mod.fc_init.in_features == 256
        assert mod.bias.shape == (3, 256, 256)
        assert isinstance(mod.nl, torch.nn.ReLU)

    @pytest.mark.parametrize(
        """batch,
            height,
            width,
            channels,
            dim_in,
            positional_bias,
            nl""",
        [
            (8, 256, 256, 3, 256, False, torch.nn.LeakyReLU()),
            (1, 128, 256, 3, 128, True, torch.nn.LeakyReLU(negative_slope=0.2)),
            (4, 512, 128, 3, 256, True, torch.nn.LeakyReLU()),
        ],
    )
    def test_forward(self, batch, height, width, channels, dim_in, positional_bias, nl):
        mod = cls(height, width, channels, dim_in, positional_bias, nl)
        x = torch.randn(batch, dim_in)
        x = mod.forward(x)
        assert x.size(0) == batch, "batch size mismatch"
        assert x.size(1) == channels, "channel size mismatch"
        assert x.size(2) == height, "height mismatch"
        assert x.size(3) == width, "width mismatch"
