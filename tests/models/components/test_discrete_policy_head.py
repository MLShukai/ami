import pytest
import torch
from torch.distributions import Categorical
from torch.distributions.distribution import Distribution

from ami.models.components.discrete_policy_head import (
    DiscretePolicyHead,
    MultiCategoricals,
)


class TestMultiCategoricals:
    @pytest.fixture
    def distributions(self) -> list[Categorical]:
        choices_per_dist = [3, 2, 5]
        batch_size = 8
        return [Categorical(logits=torch.zeros(batch_size, c)) for c in choices_per_dist]

    @pytest.fixture
    def multi_categoricals(self, distributions) -> MultiCategoricals:
        return MultiCategoricals(distributions)

    def test_init(self, multi_categoricals: MultiCategoricals):
        assert multi_categoricals.batch_shape == (8, 3)

    def test_sample(self, multi_categoricals: MultiCategoricals):
        assert multi_categoricals.sample().shape == (8, 3)
        assert multi_categoricals.sample((1, 2)).shape == (1, 2, 8, 3)

    def test_log_prob(self, multi_categoricals: MultiCategoricals):
        sampled = multi_categoricals.sample()
        assert multi_categoricals.log_prob(sampled).shape == sampled.shape

    def test_entropy(self, multi_categoricals: MultiCategoricals):
        assert multi_categoricals.entropy().shape == (8, 3)


class TestDiscretePolicyHead:
    @pytest.mark.parametrize(
        """
        batch,
        dim_in,
        action_choices_per_category,
        """,
        [
            (8, 256, [3, 3, 3, 2, 2]),
            (1, 16, [1, 2, 3, 4, 5]),
        ],
    )
    def test_discrete_policy_head(self, batch, dim_in, action_choices_per_category):
        policy = DiscretePolicyHead(dim_in, action_choices_per_category)
        input = torch.randn(batch, dim_in)
        dist = policy(input)
        assert isinstance(dist, Distribution)
        assert dist.sample().shape == (batch, len(action_choices_per_category))
        assert dist.log_prob(dist.sample()).shape == (batch, len(action_choices_per_category))
        assert dist.entropy().shape == (batch, len(action_choices_per_category))
