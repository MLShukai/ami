import pytest
import torch

from ami.models.components.multi_embeddings import MultiEmbeddings


class TestMultiEmbedding:
    @pytest.mark.parametrize(
        """
        choices_per_category,
        embedding_dim,
        """,
        [
            ([3, 4, 5], 128),
            ([5, 4, 5, 7], 123),
        ],
    )
    def test_choices_per_category(self, choices_per_category, embedding_dim):
        me = MultiEmbeddings(choices_per_category, embedding_dim)
        assert me.choices_per_category == choices_per_category

    @pytest.mark.parametrize(
        """
        batch,
        length,
        choices_per_category,
        embedding_dim,
        """,
        [
            (32, 64, [3, 4, 5], 128),
            (3, 5, [5, 4, 5, 7], 123),
        ],
    )
    def test_no_flatten(self, batch, length, choices_per_category, embedding_dim):
        me = MultiEmbeddings(choices_per_category, embedding_dim)
        input_list = []
        for choices in choices_per_category:
            input_list.append(torch.randint(choices, (batch, length)))
        input = torch.stack(input_list, dim=-1)
        output = me(input)
        assert output.shape == (batch, length, len(choices_per_category), embedding_dim)

    @pytest.mark.parametrize(
        """
        batch,
        length,
        choices_per_category,
        embedding_dim,
        """,
        [
            (32, 64, [3, 4, 5], 128),
            (3, 5, [5, 4, 5, 7], 123),
        ],
    )
    def test_do_flatten(self, batch, length, choices_per_category, embedding_dim):
        me = MultiEmbeddings(choices_per_category, embedding_dim, do_flatten=True)
        input_list = []
        for choices in choices_per_category:
            input_list.append(torch.randint(choices, (batch, length)))
        input = torch.stack(input_list, dim=-1)
        output = me(input)
        assert output.shape == (batch, length, len(choices_per_category) * embedding_dim)
