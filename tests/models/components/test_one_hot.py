import pytest
import torch
import torch.nn as nn

from ami.models.components.one_hot import (
    MaskedOneHotCategoricalStraightThrough,
    MultiOneHots,
    OneHotToEmbedding,
)


class TestMaskedOneHotCategoricalStraightThrough:
    @pytest.mark.parametrize("batch_size", [1, 32])
    @pytest.mark.parametrize("num_categories", [5])
    @pytest.mark.parametrize("num_choices", [3])
    def test_initialization(self, batch_size, num_categories, num_choices):
        logits = torch.randn(batch_size, num_categories, num_choices)
        mask = torch.zeros(num_categories, num_choices, dtype=torch.bool)
        mask[:, num_choices // 2 :] = True

        distribution = MaskedOneHotCategoricalStraightThrough(logits=logits, mask=mask)

        assert distribution.logits.shape == (batch_size, num_categories, num_choices)
        assert torch.all(distribution.logits[..., mask] == -torch.inf)
        assert distribution.probs.shape == (batch_size, num_categories, num_choices)
        assert torch.all(distribution.probs[..., mask] == 0)

    def test_entropy(self):
        logits = torch.randn(1, 3, 4)
        # fmt: off
        mask = torch.tensor(
            [
                [False, False, True, True],
                [False, False, False, True],
                [False, True, True, True],
            ]
        )
        # fmt: on

        distribution = MaskedOneHotCategoricalStraightThrough(logits=logits, mask=mask)
        entropy = distribution.entropy()

        assert entropy.shape == (1, 3)
        assert torch.all(entropy >= 0)  # Entropy should be non-negative
        assert torch.all(torch.isfinite(entropy))

    def test_log_prob(self):
        logits = torch.randn(1, 3, 4)
        # fmt: off
        mask = torch.tensor(
            [
                [False, False, True, True],
                [False, False, False, True],
                [False, True, True, True],
            ]
        )
        # fmt: on

        distribution = MaskedOneHotCategoricalStraightThrough(logits=logits, mask=mask)

        # Test with valid one-hot vectors
        # fmt: off
        valid_sample = torch.tensor(
            [
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                ]
            ]
        )
        # fmt: on
        log_prob = distribution.log_prob(valid_sample)

        assert log_prob.shape == (1, 3)
        assert torch.all(torch.isfinite(log_prob))

        # Test with invalid one-hot vectors (selecting masked values)
        # fmt: off
        invalid_sample = torch.tensor(
            [
                [
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 1, 0, 0],
                ]
            ]
        )
        # fmt: on
        log_prob_invalid = distribution.log_prob(invalid_sample)

        assert torch.all(log_prob_invalid == -torch.inf)


class TestMultiOneHots:
    @pytest.mark.parametrize("in_features", [10])
    @pytest.mark.parametrize("choices_per_category", [[2, 3, 4], [3, 3, 3, 2, 2]])
    @pytest.mark.parametrize("batch_size", [1, 32])
    def test_multi_one_hots(self, in_features, choices_per_category, batch_size):
        # Create MultiOneHots instance
        multi_one_hots = MultiOneHots(in_features, choices_per_category)

        # Create input tensor
        x = torch.randn(batch_size, in_features)

        # Forward pass
        output = multi_one_hots(x)

        # Check output type
        assert isinstance(output, MaskedOneHotCategoricalStraightThrough)

        # Check output shapes
        num_categories = len(choices_per_category)
        max_choices = max(choices_per_category)
        assert output.logits.shape == (batch_size, num_categories, max_choices)

        # Check that logits for invalid choices are set to -inf
        # Check that probs for invalid choices are set to 0
        for i, choices in enumerate(choices_per_category):
            assert torch.all(output.logits[:, i, choices:] == -torch.inf)
            assert torch.all(output.probs[:, i, choices:] == 0)

        # Test sampling
        sample = output.sample()
        assert sample.shape == (batch_size, num_categories, max_choices)
        assert torch.all(sample.sum(dim=-1) == 1)  # One-hot property

        # Test sample gradient
        assert output.rsample().requires_grad is True
        assert output.sample().requires_grad is False

    def test_multi_one_hots_invalid_args(self):
        with pytest.raises(AssertionError):
            MultiOneHots(10, [0, 1, 2])  # Category with no choices


class TestOneHotToEmbedding:
    @pytest.mark.parametrize("num_embeddings", [10])
    @pytest.mark.parametrize("embedding_dim", [5])
    @pytest.mark.parametrize("batch_size", [1, 32])
    def test_one_hot_to_embedding(self, num_embeddings, embedding_dim, batch_size):
        # Create OneHotEmbedding instance
        one_hot_embedding = OneHotToEmbedding(num_embeddings, embedding_dim)

        # Create input tensor (one-hot vectors)
        x = torch.eye(num_embeddings).repeat(batch_size, 1, 1)

        # Forward pass
        output = one_hot_embedding(x)

        # Check output shape
        assert output.shape == (batch_size, num_embeddings, embedding_dim)

        # Check that the embedding for each one-hot vector is correct
        for i in range(num_embeddings):
            assert torch.allclose(output[:, i], one_hot_embedding._weight[i])

    def test_one_hot_to_embedding_gradients(self):
        num_embeddings, embedding_dim = 10, 5
        one_hot_embedding = OneHotToEmbedding(num_embeddings, embedding_dim)
        x = torch.eye(num_embeddings)

        output = one_hot_embedding(x)

        # Check that gradients
        assert one_hot_embedding._weight.requires_grad is True
        assert output.requires_grad is True
