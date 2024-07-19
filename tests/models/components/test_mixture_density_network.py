import pytest
import torch
from torch.distributions import Normal

from ami.models.components.mixture_desity_network import NormalMixture


class TestNormalMixture:
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    @pytest.mark.parametrize("num_components", [2, 3])
    def test_normal_mixture(self, batch_shape, num_components):
        # Create test data
        log_pi = torch.randn(*batch_shape, num_components).log_softmax(-1)
        mu = torch.randn(*batch_shape, num_components)
        sigma = torch.rand(*batch_shape, num_components).add_(0.1)  # Ensure positive values

        # Create NormalMixture instance
        mixture = NormalMixture(log_pi, mu, sigma)

        # Test batch_shape
        assert mixture.batch_shape == torch.Size(batch_shape)

        # Test sampling
        sample = mixture.sample()
        assert sample.shape == torch.Size(batch_shape)

        # Test log_prob
        log_prob = mixture.log_prob(sample)
        assert log_prob.shape == torch.Size(batch_shape)

        sample_shape = (10, 5)
        samples = mixture.sample(sample_shape)
        assert samples.shape == torch.Size(sample_shape + batch_shape)
        assert mixture.log_prob(samples).shape == sample_shape + batch_shape

        # Test rsample
        rsample = mixture.rsample()
        assert rsample.shape == torch.Size(batch_shape)

        # Test consistency with individual normal components
        components = [Normal(mu[..., i], sigma[..., i]) for i in range(num_components)]
        mixture_log_prob = mixture.log_prob(sample)
        component_log_probs = torch.stack([comp.log_prob(sample) for comp in components], dim=-1)
        component_log_probs += log_pi
        expected_log_prob = torch.logsumexp(component_log_probs, dim=-1)
        assert torch.allclose(mixture_log_prob, expected_log_prob, atol=1e-5)

    def test_normal_mixture_invalid_args(self):
        # Test error handling for invalid arguments
        with pytest.raises(AssertionError):
            NormalMixture(torch.randn(3, 2), torch.randn(3, 3), torch.rand(3, 2).add_(0.1))
