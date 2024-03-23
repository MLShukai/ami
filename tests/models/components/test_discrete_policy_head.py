import pytest
import torch
from torch.distributions import Categorical

from ami.models.components.discrete_policy_head import MultiCategoricals


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
