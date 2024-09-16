import torch

from ami.models.components.stacked_features import (
    LerpStackedFeatures,
    NormalMixture,
    ToStackedFeatures,
    ToStackedFeaturesMDN,
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


class TestToStackedFeaturesMDN:
    def test_forward(self):
        mod = ToStackedFeaturesMDN(64, 128, 4, 8)

        feature = torch.randn(8, 64)
        out = mod.forward(feature)
        assert isinstance(out, NormalMixture)
        assert out.sample().shape == (8, 4, 128)
        assert out.num_components == 8
        assert torch.allclose(out.logits, torch.zeros_like(out.logits), atol=0.5)
        assert torch.allclose(out.sigma, torch.ones_like(out.sigma), atol=0.5)

        feature = torch.randn(64)
        out = mod.forward(feature)
        assert out.sample().shape == (4, 128)

        feature = torch.randn(3, 8, 64)
        out = mod.forward(feature)
        assert out.sample().shape == (3, 8, 4, 128)
