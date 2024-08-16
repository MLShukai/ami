import torch

from ami.models.components.stacked_features import LerpStackedFeatures


class TestLerpedStackedFeatures:
    def test_forward(self):
        mod = LerpStackedFeatures(128, 64, 8, 4)

        hidden = torch.randn(4, 8, 128)
        out = mod.forward(hidden)
        assert out.shape == (4, 64)
