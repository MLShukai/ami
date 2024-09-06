import pytest
import torch

from ami.models.components.deterministic import Deterministic


class TestDeterministic:
    @pytest.fixture(params=[(), (2,), (2, 3), (2, 3, 4)])
    def data(self, request):
        return torch.randn(request.param)

    @pytest.fixture
    def deterministic(self, data):
        return Deterministic(data)

    def test_init(self, data, deterministic):
        assert torch.all(deterministic.data == data)
        assert deterministic.event_shape == torch.Size(data.shape)

    def test_rsample(self, data, deterministic):
        rsample = deterministic.rsample()
        assert rsample.shape == data.shape
        assert torch.allclose(rsample, data)

        sample_shape = (10, 5)
        rsamples = deterministic.rsample(sample_shape)
        assert rsamples.shape == torch.Size(sample_shape + data.shape)
        assert torch.all(rsamples == data.expand(*sample_shape, *data.shape))

    def test_entropy(self, data, deterministic):
        entropy = deterministic.entropy()
        assert entropy.shape == data.shape
        assert torch.all(entropy == 0)

    def test_log_prob(self, data, deterministic):
        log_prob = deterministic.log_prob(data)
        assert log_prob.shape == data.shape
        assert torch.all(log_prob == 0)

        different_data = data + 1
        log_prob_diff = deterministic.log_prob(different_data)
        assert torch.all(log_prob_diff == float("-inf"))
