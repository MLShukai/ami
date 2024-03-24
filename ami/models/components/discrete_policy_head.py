import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution


class MultiCategoricals(Distribution):
    """Set of same action torch.Size categorical distributions."""

    arg_constraints = {}

    def __init__(self, distributions: list[Categorical]) -> None:
        """Constructs Multi Categorical class. All `batch_shape` of child
        categorical class must be same.

        Args:
            distributions: A list of Categorical distributions, where each distribution may have a different size of action choices.
        """

        assert len(distributions) > 0
        first_dist = distributions[0]
        assert all(first_dist.batch_shape == d.batch_shape for d in distributions), "All batch shapes must be same."

        batch_shape = torch.Size((*first_dist.batch_shape, len(distributions)))
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size(), validate_args=False)

        self.dists = distributions

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample from each distributions and stacks their outputs.

        Shape:
            return: (*sample_shape, num_dists)
        """
        return torch.stack([d.sample(sample_shape) for d in self.dists], dim=-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability of `value`

        Shape:
            value: (*, num_dists)
            return: (*, num_dists)
        """
        return torch.stack([d.log_prob(v) for d, v in zip(self.dists, value.movedim(-1, 0))], dim=-1)

    def entropy(self) -> torch.Tensor:
        """Compute entropy for each distribution."""
        return torch.stack([d.entropy() for d in self.dists], dim=-1)


class DiscretePolicyHead(nn.Module):
    """Policy head for discrete action space."""

    def __init__(self, dim_in: int, action_choices_per_category: list[int]) -> None:
        """Constructs policy.

        Args:
            dim_in: Input dimension size of tensor.
            action_choices_per_category: List of action choice count per category.
        """
        super().__init__()

        self.heads = nn.ModuleList()
        for choice in action_choices_per_category:
            self.heads.append(nn.Linear(dim_in, choice, bias=False))

    def forward(self, input: torch.Tensor) -> MultiCategoricals:

        categoricals = []
        for head in self.heads:
            logits = head(input)
            categoricals.append(Categorical(logits=logits))

        return MultiCategoricals(categoricals)
