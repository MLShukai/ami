import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor
from torch.distributions import Distribution, Normal, constraints


class NormalMixture(Distribution):
    """Computes Mixture Density Distribution of Normal distributions."""

    SQRT_2_PI = (2 * torch.pi) ** 0.5
    arg_constraints = {
        "logits": constraints.real,
        "mu": constraints.real,
        "sigma": constraints.positive,
    }

    def __init__(
        self, logits: Tensor, mu: Tensor, sigma: Tensor, eps: float = 1e-6, validate_args: bool | None = None
    ) -> None:
        """Constructor for the NormalMixture class.

        This constructor initializes the parameters of the mixture normal distribution and calls the parent class constructor.
        logits, mu, sigma are must be same shape.

        Args:
            logits: Tensor representing the unnormalized log probabilities for each component in the mixture distribution.
            mu: Tensor representing the means of each normal distribution.
            sigma: Tensor representing the standard deviations of each normal distribution.
            eps: A small value for numerical stability.
            validate_args: Whether to validate the arguments.
        Shape:
            logits, mu, sigma: (*, Components)
        """
        assert logits.shape == mu.shape == sigma.shape
        batch_shape = logits.shape[:-1]
        self.num_components = logits.size(-1)
        self.logits = logits
        self.mu = mu
        self.sigma = sigma
        self.eps = eps

        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def log_pi(self) -> Tensor:
        return self.logits.log_softmax(-1)

    def _get_expand_shape(self, shape: Size) -> tuple[int, ...]:
        return *shape, *self.batch_shape, self.num_components

    def rsample(self, sample_shape: Size = Size(), temperature: float = 1.0) -> Tensor:
        """
        Args:
            temperature: Sampling uncertainty.
        """
        shape = self._get_expand_shape(sample_shape)

        pi = self.logits.div(temperature).softmax(-1).expand(shape).contiguous()
        samples = torch.multinomial(
            pi.view(-1, pi.size(-1)),
            1,
        ).view(*pi.shape[:-1], 1)
        sample_mu = self.mu.expand(shape).gather(-1, samples).squeeze(-1)
        sample_sigma = self.sigma.expand(shape).gather(-1, samples).squeeze(-1)
        return torch.randn_like(sample_mu) * sample_sigma + sample_mu

    @torch.no_grad()
    def sample(self, sample_shape: Size = Size(), temperature: float = 1.0) -> Tensor:
        return self.rsample(sample_shape, temperature)

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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_components: int,
        squeeze_feature_dim: bool = False,
        eps: float = 1e-4,
    ) -> None:
        """
        Args:
            in_feature: The number of input features.
            out_features: The number of output features (dimensionality of each normal distribution).
            num_components: The number of mixture components.
            squeeze_feature_dim: Whether or not to squeeze the feature dimension of output tensor.
            eps: A small value for numerical stability.
        """
        super().__init__()
        if squeeze_feature_dim:
            assert out_features == 1, "Can not squeeze feature dimension!"
        self.mu_layers = nn.ModuleList(nn.Linear(in_features, out_features) for _ in range(num_components))
        self.logvar_layers = nn.ModuleList(nn.Linear(in_features, out_features) for _ in range(num_components))
        self.logits_layers = nn.ModuleList(nn.Linear(in_features, out_features) for _ in range(num_components))

        self.squeeze_feature_dim = squeeze_feature_dim
        self.eps = eps

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize the weights.

        The sigma layer will output to 1, and logits layer will output
        uniform distribution.
        """
        layer: nn.Linear
        for layer in self.logvar_layers:
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.zeros_(layer.bias)

        for layer in self.logits_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> NormalMixture:
        mu = torch.stack([lyr(x) for lyr in self.mu_layers], dim=-1)
        sigma = torch.stack([torch.exp(0.5 * lyr(x)) for lyr in self.logvar_layers], dim=-1) + self.eps
        logits = torch.stack([lyr(x) for lyr in self.logits_layers], dim=-1)

        if self.squeeze_feature_dim:
            mu = mu.squeeze(-2)
            sigma = sigma.squeeze(-2)
            logits = logits.squeeze(-2)

        return NormalMixture(logits, mu, sigma, self.eps)
