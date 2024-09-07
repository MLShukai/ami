import torch
from torch import Size, Tensor
from torch.distributions import Distribution


class Deterministic(Distribution):
    """Deterministic distribution class for treating pure tensor as a
    distribution.

    This class represents a deterministic distribution, which always
    returns the same value (the provided data) when sampled. It can be
    useful in scenarios where you want to treat a fixed tensor as a
    distribution, maintaining compatibility with other distribution-
    based APIs.
    """

    def __init__(self, data: Tensor):
        """Constructor for the Deterministic class."""
        self.data = data
        super().__init__(Size(), data.shape, False)

    def rsample(self, sample_shape: Size = Size()) -> Tensor:
        """This method always returns the same data, expanded to the requested
        sample shape."""
        shape = self._extended_shape(sample_shape)
        return self.data.expand(shape)

    def entropy(self) -> Tensor:
        """Computes the entropy of the deterministic distribution.

        The entropy of a deterministic distribution is always zero.
        """
        return torch.zeros_like(self.data)

    def log_prob(self, value: Tensor) -> Tensor:
        """Computes the log probability of the given value.

        For a deterministic distribution, the log probability is 0
        (probability 1) if the value exactly matches the data, and
        negative infinity (probability 0) otherwise.
        """
        prob = torch.zeros_like(value)
        prob[self.data != value] = -torch.inf
        return prob
