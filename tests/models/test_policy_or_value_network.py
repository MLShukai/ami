import pytest
import torch
import torch.nn as nn

from ami.models.components.fully_connected_normal import FullyConnectedNormal
from ami.models.policy_or_value_network import PolicyOrValueNetwork


class CatObsHidden(nn.Module):
    def forward(self, obs, hidden):
        return torch.cat([obs, hidden], -1)


class TestPolicyOrValueNetwork:
    @pytest.fixture
    def net(self) -> PolicyOrValueNetwork:
        obs_layer = nn.Linear(128, 64)
        hidden_layer = nn.Linear(256, 128)
        obs_hidden_proj = CatObsHidden()
        core_model = nn.Linear(128 + 64, 16)
        head = FullyConnectedNormal(16, 8)
        return PolicyOrValueNetwork(obs_layer, hidden_layer, obs_hidden_proj, core_model, head)

    def test_forward(self, net: PolicyOrValueNetwork):
        dist = net.forward(torch.randn(128), torch.randn(256))
        assert dist.sample().shape == (8,)
