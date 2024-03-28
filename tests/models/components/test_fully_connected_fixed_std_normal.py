import torch
from torch.distributions import Normal

from ami.models.components.fully_connected_fixed_std_normal import (
    DeterministicNormal,
    FullyConnectedFixedStdNormal,
)


class TestFullyConnectedFixedStdNormal:
    def test_forward(self):
        layer = FullyConnectedFixedStdNormal(10, 20)
        out = layer(torch.randn(10))

        assert isinstance(out, Normal)
        assert out.sample().shape == (20,)

        assert layer(torch.randn(1, 2, 3, 10)).sample().shape == (1, 2, 3, 20)


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
