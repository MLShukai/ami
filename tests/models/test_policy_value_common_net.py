import pytest
import torch
import torch.nn as nn
from torch.distributions import Distribution

from ami.models.components.discrete_policy_head import DiscretePolicyHead
from ami.models.components.fully_connected_value_head import FullyConnectedValueHead
from ami.models.policy_value_common_net import (
    ConcatFlattenedObservationAndLerpedHidden,
    ConcatFlattenedObservationAndStackedHidden,
    LerpStackedHidden,
    PolicyValueCommonNet,
    PrimitivePolicyValueCommonNet,
)


class CatObsHidden(nn.Module):
    def forward(self, obs, hidden):
        return torch.cat([obs, hidden], -1)


class TestPolicyValueCommonNet:
    @pytest.fixture
    def net(self) -> PolicyValueCommonNet:
        obs_layer = nn.Linear(128, 64)
        hidden_layer = nn.Linear(256, 128)
        obs_hidden_proj = CatObsHidden()
        core_model = nn.Linear(128 + 64, 16)
        policy = DiscretePolicyHead(16, [8])
        value = FullyConnectedValueHead(16)

        return PolicyValueCommonNet(obs_layer, hidden_layer, obs_hidden_proj, core_model, policy, value)

    def test_forward(self, net: PolicyValueCommonNet):

        action_dist, value = net.forward(torch.randn(128), torch.randn(256))
        assert isinstance(action_dist, Distribution)
        assert action_dist.sample().shape == (1,)
        assert value.shape == (1,)


class TestConcatFlattenedObservationAndStackedHidden:
    def test_forward(self):
        mod = ConcatFlattenedObservationAndStackedHidden()

        obs, hidden = torch.randn(3, 10), torch.randn(3, 4, 10)
        out = mod.forward(obs, hidden)
        assert out.shape == (3, 5, 10)
        assert torch.equal(obs, out[:, 0])

        mod = ConcatFlattenedObservationAndStackedHidden(transpose=True)
        out = mod.forward(obs, hidden)
        assert out.shape == (3, 10, 5)
        assert torch.equal(obs, out[:, :, 0])


class TestLerpedStackedHidden:
    def test_forward(self):
        mod = LerpStackedHidden(128, 8, 4)

        hidden = torch.randn(4, 8, 128)
        out = mod.forward(hidden)
        assert out.shape == (4, 128)

        hidden = torch.randn(8, 128)
        out = mod.forward(hidden)
        assert out.shape == (128,)


class TestConcatFlattenedObservationAndLerpedHidden:
    def test_forward(self):
        mod = ConcatFlattenedObservationAndLerpedHidden(32, 64, 128)

        obs = torch.randn(4, 32)
        hidden = torch.randn(4, 64)
        out = mod.forward(obs, hidden)
        assert out.shape == (4, 128)


class TestPrimitivePolicyValueCommonNet:
    @pytest.fixture
    def net(self) -> PrimitivePolicyValueCommonNet:
        obs_layer = nn.Linear(128, 64)
        core_model = nn.Linear(64, 16)
        policy = DiscretePolicyHead(16, [8])
        value = FullyConnectedValueHead(16)

        return PrimitivePolicyValueCommonNet(obs_layer, core_model, policy, value)

    def test_forward(self, net: PrimitivePolicyValueCommonNet):

        action_dist, value = net.forward(torch.randn(128))
        assert isinstance(action_dist, Distribution)
        assert action_dist.sample().shape == (1,)
        assert value.shape == (1,)
