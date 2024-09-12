import torch

from ami.models.components.stacked_features import (
    LerpStackedFeatures,
    ToStackedFeatures,
)


class TestLerpedStackedFeatures:
    def test_forward(self):
        mod = LerpStackedFeatures(128, 64, 8)

        feature = torch.randn(4, 8, 128)
        out = mod.forward(feature)
        assert out.shape == (4, 64)

        feature = torch.randn(3, 4, 8, 128)
        out = mod.forward(feature)
        assert out.shape == (3, 4, 64)


class TestToStackedFeatures:
    def test_forward(self):
        mod = ToStackedFeatures(64, 128, 4)

        feature = torch.randn(8, 64)
        out = mod.forward(feature)
        assert out.shape == (8, 4, 128)

        feature = torch.randn(64)
        out = mod.forward(feature)
        assert out.shape == (4, 128)

        feature = torch.randn(3, 8, 64)
        out = mod.forward(feature)
        assert out.shape == (3, 8, 4, 128)
