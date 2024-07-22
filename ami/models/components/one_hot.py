from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import OneHotCategoricalStraightThrough


class MaskedOneHotCategoricalStraightThrough(OneHotCategoricalStraightThrough):
    """Masked OneHotCategoricalStraightThrough class.

    This class extends OneHotCategoricalStraightThrough by adding
    functionality to apply a mask, disabling specific choices.
    """

    def __init__(self, logits: Tensor, mask: Tensor, validate_args: Any = None) -> None:
        """
        Args:
            logits: Tensor of logits.
            mask: Mask tensor. True elements indicate invalid choices.
        """
        assert logits.shape[-mask.ndim :] == mask.shape
        logits[..., mask] = -torch.inf
        self.mask = mask
        super().__init__(logits=logits, validate_args=validate_args)

    def entropy(self) -> Tensor:
        log_prob = torch.log_softmax(self.logits, dim=-1)
        log_prob[..., self.mask] = 0.0
        return -torch.sum(self.probs * log_prob, dim=-1)


class MultiOneHots(nn.Module):
    """Class for handling multiple OneHot distributions.

    This class generates OneHot distributions for multiple categories.
    Each category can have a different number of choices.
    """

    def __init__(self, in_features: int, choices_per_category: list[int]) -> None:
        """
        Args:
            in_features: Number of input features.
            choices_per_category: List of number of choices for each category.
        """
        super().__init__()

        out_features = max(choices_per_category)
        num_cateogies = len(choices_per_category)
        mask = torch.zeros((num_cateogies, out_features), dtype=torch.bool)
        for i, c in enumerate(choices_per_category):
            assert c > 0, f"Category index {i} has no choices!"
            mask[i, c:] = True

        self.mask: Tensor
        self.register_buffer("mask", mask)

        self._layers = nn.ModuleList(nn.Linear(in_features, out_features, bias=False) for _ in choices_per_category)

    def forward(self, x: Tensor) -> MaskedOneHotCategoricalStraightThrough:
        """
        Shapes:
            x: (*, in_features)
            return: (*, num_categories, max_choice)
        """
        out = torch.stack([lyr(x) for lyr in self._layers], dim=-2)
        return MaskedOneHotCategoricalStraightThrough(logits=out, mask=self.mask)


class OneHotToEmbedding(nn.Module):
    """Make the embedding from one hot vectors."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self._weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim), True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x: (*, num_embeddings)
            return: (*, embedding_dim)
        """
        return x @ self._weight
