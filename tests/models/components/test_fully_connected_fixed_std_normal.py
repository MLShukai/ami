import pytest
import torch
from torch.distributions import Normal

from ami.models.components.fully_connected_fixed_std_normal import (
    SCALE_ONE,
    SHIFT_ZERO,
    DeterministicNormal,
    FullyConnectedFixedStdNormal,
)


def test_logprob_shiftzero():
    # 負の対数尤度の計算時のシフト(最小値)が二乗誤差と等しい場合のテスト。
    # スケールは `math.pi`だけズレる。

    mean = torch.zeros(10)
    std = torch.full_like(mean, SHIFT_ZERO)
    normal = Normal(mean, std)
    expected = torch.zeros_like(mean)

    torch.testing.assert_close(normal.log_prob(mean), expected)


def test_logprob_scaleone():
    # 負の対数尤度の計算時のスケールが二乗誤差と等しい場合のテスト。
    # 誤差のシフトが `0.5 * math.log(math.pi)` だけ異なる。
    mean = torch.zeros(3)
    std = torch.full_like(mean, SCALE_ONE)
    normal = Normal(mean, std)

    t1 = torch.full_like(mean, 1)
    t2 = torch.full_like(mean, 3)

    nlp1 = -normal.log_prob(t1)
    nlp2 = -normal.log_prob(t2)

    expected = (t1 - mean) ** 2 - (t2 - mean) ** 2
    actual = nlp1 - nlp2

    torch.testing.assert_close(actual, expected)


class TestFullyConnectedFixedStdNormal:
    def test_forward(self):
        layer = FullyConnectedFixedStdNormal(10, 20)
        out = layer(torch.randn(10))

        assert isinstance(out, Normal)
        assert out.sample().shape == (20,)

        assert layer(torch.randn(1, 2, 3, 10)).sample().shape == (1, 2, 3, 20)

    def test_squeeze_feature_dim(self):
        with pytest.raises(AssertionError):
            # out_features must be 1.
            FullyConnectedFixedStdNormal(10, 2, squeeze_feature_dim=True)

        # `squeeze_feature_dim` default false.
        FullyConnectedFixedStdNormal(10, 2)

        net = FullyConnectedFixedStdNormal(10, 1, squeeze_feature_dim=True)
        x = torch.randn(10)
        out = net(x)
        assert out.sample().shape == ()


class TestDeterministicNormal:
    def test_sample(self):
        mean = torch.randn(10)
        std = torch.ones(10)
        dn = DeterministicNormal(mean, std)
        assert torch.equal(dn.sample(), mean)
        assert torch.equal(dn.sample(), mean)
        assert torch.equal(dn.sample(), mean)
        assert torch.equal(dn.rsample(), mean)
        assert torch.equal(dn.rsample(), mean)
        assert torch.equal(dn.rsample(), mean)
