import pytest
import torch
import torch.nn as nn
from torch.distributions import Distribution

from ami.models.components.discrete_policy_head import DiscretePolicyHead
from ami.models.components.fully_connected_value_head import FullyConnectedValueHead
from ami.models.policy_value_common_net import ModularPolicyValueCommonNet


class TestModularPolicyValueCommonNet:
    @pytest.fixture
    def net(self) -> ModularPolicyValueCommonNet:
        base_model = nn.Linear(128, 16)
        policy = DiscretePolicyHead(16, [8])
        value = FullyConnectedValueHead(16)

        return ModularPolicyValueCommonNet(base_model, policy, value)

    def test_forward(self, net: ModularPolicyValueCommonNet):

        action_dist, value = net.forward(torch.randn(128))
        assert isinstance(action_dist, Distribution)
        assert action_dist.sample().shape == (1,)
        assert value.shape == (1,)
