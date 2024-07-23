import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor
from torch.distributions import Distribution, Normal, constraints


class NormalMixture(Distribution):
    """Computes Mixture Density Distribution of Normal distributions."""

    SQRT_2_PI = (2 * torch.pi) ** 0.5
    arg_constraints = {
        "log_pi": constraints.less_than(0.0),
        "mu": constraints.real,
        "sigma": constraints.positive,
    }

    def __init__(
        self, log_pi: Tensor, mu: Tensor, sigma: Tensor, eps: float = 1e-6, validate_args: bool | None = None
    ) -> None:
        """Constructor for the NormalMixture class.

        This constructor initializes the parameters of the mixture normal distribution and calls the parent class constructor.
        log_pi, mu, sigma are must be same shape.

        Args:
            log_pi: Tensor representing the mixture log ratios of each normal distribution.
            mu: Tensor representing the means of each normal distribution.
            sigma: Tensor representing the standard deviations of each normal distribution.
            eps: A small value for numerical stability.
            validate_args: Whether to validate the arguments.
        Shape:
            log_pi, mu, sigma: (*, Components)
        """
        assert log_pi.shape == mu.shape == sigma.shape
        batch_shape = log_pi.shape[:-1]
        self.num_components = log_pi.size(-1)
        self.log_pi = log_pi
        self.mu = mu
        self.sigma = sigma
        self.eps = eps

        super().__init__(batch_shape, validate_args=validate_args)

    def _get_expand_shape(self, shape: Size) -> tuple[int, ...]:
        return *shape, *self.batch_shape, self.num_components

    def rsample(self, sample_shape: Size = Size()) -> Tensor:
        shape = self._get_expand_shape(sample_shape)

        pi = self.log_pi.exp().expand(shape).contiguous()
        samples = torch.multinomial(
            pi.view(-1, pi.size(-1)),
            1,
        ).view(*pi.shape[:-1], 1)
        sample_mu = self.mu.expand(shape).gather(-1, samples).squeeze(-1)
        sample_sigma = self.sigma.expand(shape).gather(-1, samples).squeeze(-1)
        return torch.randn_like(sample_mu) * sample_sigma + sample_mu

    def sample(self, sample_shape: Size = Size()) -> Tensor:
        return self.rsample(sample_shape).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        shape = *value.shape, self.num_components
        mu = self.mu.expand(shape)
        sigma = self.sigma.expand(shape)
        log_pi = self.log_pi.expand(shape)
        normal_prob = -0.5 * ((value.unsqueeze(-1) - mu) / (sigma + self.eps)) ** 2 - torch.log(
            self.SQRT_2_PI * sigma + self.eps
        )
        return torch.logsumexp(log_pi + normal_prob, -1)


class NormalMixtureDensityNetwork(nn.Module):
    """A neural network that outputs parameters for a mixture of normal
    distributions.

    This network takes an input tensor and produces the parameters
    (mixture weights, means, and standard deviations) for a mixture of
    normal distributions. It can be used as the output layer in a neural
    network for tasks that require modeling complex, multi-modal
    distributions.
    """

    def __init__(self, in_features: int, out_features: int, num_components: int) -> None:
        """
        Args:
            in_feature: The number of input features.
            out_features: The number of output features (dimensionality of each normal distribution).
            num_components: The number of mixture components.
        """
        super().__init__()
        self.mu_layers = nn.ModuleList(nn.Linear(in_features, out_features) for _ in range(num_components))
        self.sigma_layers = nn.ModuleList(nn.Linear(in_features, out_features) for _ in range(num_components))
        self.logits_layers = nn.ModuleList(nn.Linear(in_features, out_features) for _ in range(num_components))

    def forward(self, x: Tensor) -> NormalMixture:
        mu = torch.stack([lyr(x) for lyr in self.mu_layers], dim=-1)
        sigma = torch.stack([F.softplus(lyr(x)) for lyr in self.sigma_layers], dim=-1)
        log_pi = torch.stack([lyr(x) for lyr in self.logits_layers], dim=-1).log_softmax(-1)
        return NormalMixture(log_pi, mu, sigma)
