import pytest
import torch
import torch.nn as nn
from torch.distributions import Normal

from ami.models.components.mixture_desity_network import (
    NormalMixture,
    NormalMixtureDensityNetwork,
)


class TestNormalMixture:
    @pytest.mark.parametrize("batch_shape,event_shape", [[(), (3,)], [(2,), ()], [(2, 3), (4, 5)]])
    @pytest.mark.parametrize("num_components", [2, 3])
    def test_normal_mixture(self, batch_shape, event_shape, num_components):
        # Create test data
        logits = torch.randn(*batch_shape, num_components)
        mu = torch.randn(*batch_shape, *event_shape, num_components)
        sigma = torch.rand(*batch_shape, *event_shape, num_components).add_(0.1)  # Ensure positive values
        shape = *batch_shape, *event_shape
        # Create NormalMixture instance
        mixture = NormalMixture(logits, mu, sigma)

        # Test batch_shape and event_shape
        assert mixture.batch_shape == torch.Size(batch_shape)
        assert mixture.event_shape == torch.Size(event_shape)

        # Test sampling
        sample = mixture.sample()
        assert sample.shape == torch.Size(shape)

        # Test log_prob
        log_prob = mixture.log_prob(sample)
        assert log_prob.shape == torch.Size(shape)

        sample_shape = (10, 5)
        samples = mixture.sample(sample_shape)
        assert samples.shape == torch.Size(sample_shape + shape)
        assert mixture.log_prob(samples).shape == sample_shape + shape

        # Test rsample
        rsample = mixture.rsample()
        assert rsample.shape == torch.Size(shape)

        # Test temperature sample
        assert mixture.rsample(temperature=10.0).shape == torch.Size(shape)
        assert mixture.sample(temperature=0.1).shape == torch.Size(shape)

        # Test consistency with individual normal components
        components = [Normal(mu[..., i], sigma[..., i]) for i in range(num_components)]
        mixture_log_prob = mixture.log_prob(sample)
        component_log_probs = torch.stack([comp.log_prob(sample) for comp in components], dim=-1)
        for _ in range(mu.ndim - logits.ndim):
            logits.unsqueeze_(-2)
        component_log_probs += logits.log_softmax(-1)
        expected_log_prob = torch.logsumexp(component_log_probs, dim=-1)
        assert torch.allclose(mixture_log_prob, expected_log_prob, atol=1e-3)

    def test_normal_mixture_invalid_args(self):
        # Test error handling for invalid arguments
        with pytest.raises(AssertionError):
            NormalMixture(torch.randn(3, 2), torch.randn(3, 3), -torch.ones(3, 2))


class TestNormalMixtureDensityNetwork:
    @pytest.mark.parametrize("in_features", [10])
    @pytest.mark.parametrize("out_features", [5])
    @pytest.mark.parametrize("num_components", [2])
    @pytest.mark.parametrize("batch_size", [1, 32])
    def test_normal_mixture_density_network(self, in_features, out_features, num_components, batch_size):
        # Create NormalMixtureDensityNetwork instance
        network = NormalMixtureDensityNetwork(in_features, out_features, num_components)

        # Create input tensor
        x = torch.randn(batch_size, in_features)

        # Forward pass
        output = network(x)

        # Check output type
        assert isinstance(output, NormalMixture)

        # Check output shapes
        assert output.batch_shape == torch.Size([batch_size, out_features])
        assert output.event_shape == torch.Size([])
        assert output.logits.shape == (batch_size, out_features, num_components)
        assert output.log_pi.shape == (batch_size, out_features, num_components)
        assert output.mu.shape == (batch_size, out_features, num_components)
        assert output.sigma.shape == (batch_size, out_features, num_components)

        # Check that sigma is positive
        assert (output.sigma > 0).all()

        # Check that log_pi is a valid log probability
        assert torch.allclose(output.log_pi.exp().sum(dim=-1), torch.ones(batch_size, out_features))

        # Check the initial outputs
        assert torch.allclose(output.sigma, torch.nn.functional.softplus(torch.ones_like(output.sigma)) + output.eps)
        assert torch.allclose(output.logits, torch.zeros_like(output.logits))

    def test_normal_mixture_density_network_gradients(self):
        in_features, out_features, num_components = 10, 5, 3
        network = NormalMixtureDensityNetwork(in_features, out_features, num_components)
        x = torch.randn(32, in_features, requires_grad=True)

        output = network(x)
        sample = output.rsample()
        loss = sample.sum()
        loss.backward()

        # Check that gradients are computed
        assert x.grad is not None
        assert output.sample().grad is None

    def test_squeeze_feature_dim(self):
        with pytest.raises(AssertionError):
            # out_features must be 1.
            NormalMixtureDensityNetwork(10, 2, 3, squeeze_feature_dim=True)

        # `squeeze_feature_dim` default false.
        NormalMixtureDensityNetwork(10, 2, 3)

        # check squeezing.
        net = NormalMixtureDensityNetwork(10, 1, 3, squeeze_feature_dim=True)
        x = torch.randn(10)
        out = net(x)
        assert out.sample().shape == ()
