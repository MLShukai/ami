import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution


class PolicyValueCommonNet(nn.Module):
    """Modules with shared models for policy and value functions."""

    def __init__(self, core_model: nn.Module, policy_head: nn.Module, value_head: nn.Module) -> None:
        super().__init__()
        self.core_model = core_model
        self.policy_head = policy_head
        self.value_head = value_head

    def forward(self, observation: Tensor) -> tuple[Distribution, Tensor]:
        """Returns the action distribution and estimated value."""
        h = self.core_model(observation)
        return self.policy_head(h), self.value_head(h)
