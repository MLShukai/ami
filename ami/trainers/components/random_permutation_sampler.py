from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Sampler


class RandomPermutationSampler(Sampler[list[int]]):
    """A sampler that randomly samples sequences from a permuted time series
    dataset.

    Each sample contains a list of indices of specified length, starting
    from a randomly selected position in the entire dataset.
    """

    def __init__(self, dataset: Dataset[Any], sequence_length: int, max_samples: int | None = None) -> None:
        """
        Args:
            dataset (Dataset): The dataset to sample from
            sequence_length (int): Length of each sequence.
            max_samples (int | None): Maximum number of samples to generate
        """
        assert hasattr(dataset, "__len__")
        assert len(dataset) >= sequence_length
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.max_samples = len(self.dataset) - self.sequence_length + 1 if max_samples is None else max_samples

    def __len__(self) -> int:
        """Returns the total number of batches generated by the sampler."""
        return min(len(self.dataset) - self.sequence_length + 1, self.max_samples)

    def __iter__(self) -> Iterator[list[int]]:
        """Returns an iterator producing sequences of permuted indices.

        Each sequence starts at a random position in the permuted
        dataset and has a length of sequence_length.
        """
        start_indices = torch.randperm(len(self.dataset) - self.sequence_length + 1)[: len(self)].tolist()
        perm_indices = torch.randperm(len(self.dataset))
        for start in start_indices:
            yield perm_indices[start : start + self.sequence_length].tolist()
