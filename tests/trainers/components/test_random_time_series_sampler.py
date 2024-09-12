import pytest
import torch
from torch.utils.data import TensorDataset

from ami.trainers.components.random_time_series_sampler import (  # Assume the class is in 'your_module'
    RandomTimeSeriesSampler,
)


class TestRandomTimeSeriesSampler:
    @pytest.fixture
    def sample_dataset(self):
        data = torch.arange(100)
        return TensorDataset(data)

    def test_init(self, sample_dataset):
        sampler = RandomTimeSeriesSampler(sample_dataset, sequence_length=10, max_samples=5)
        assert sampler.sequence_length == 10
        assert sampler.max_samples == 5

        # Test assertion for dataset length
        with pytest.raises(AssertionError):
            RandomTimeSeriesSampler(TensorDataset(torch.arange(5)), sequence_length=10, max_samples=5)

    def test_len(self, sample_dataset):
        # Test when max_samples is the limiting factor
        sampler1 = RandomTimeSeriesSampler(sample_dataset, sequence_length=10, max_samples=5)
        assert len(sampler1) == 5

        # Test when dataset length is the limiting factor
        sampler2 = RandomTimeSeriesSampler(sample_dataset, sequence_length=10, max_samples=1000)
        assert len(sampler2) == 91  # 100 - 10 + 1 = 90

    def test_iter(self, sample_dataset):
        sampler = RandomTimeSeriesSampler(sample_dataset, sequence_length=5, max_samples=10)
        batches = list(iter(sampler))

        assert len(batches) == 10
        for batch in batches:
            assert len(batch) == 5  # sequence_length
            assert all(0 <= idx < 100 for idx in batch)
            assert batch[-1] - batch[0] == 4  # Check if indices are consecutive
