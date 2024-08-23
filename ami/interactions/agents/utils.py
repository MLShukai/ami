from typing import Any

from torch import Tensor
from torch.distributions import Distribution

from ami.models.model_wrapper import ThreadSafeInferenceWrapper
from ami.models.policy_or_value_network import PolicyOrValueNetwork


class PolicyValueCommonProxy:
    """Proxy class of `PolicyValueCommonNet` for the isolated policy and
    value."""

    def __init__(
        self,
        policy_net: ThreadSafeInferenceWrapper[PolicyOrValueNetwork],
        value_net: ThreadSafeInferenceWrapper[PolicyOrValueNetwork],
    ) -> None:
        self.policy_net = policy_net
        self.value_net = value_net

    def __call__(self, *args: Any, **kwds: Any) -> tuple[Distribution, Tensor]:
        return self.policy_net(*args, **kwds), self.value_net(*args, **kwds).sample()
