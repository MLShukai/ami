import random

import numpy as np
import pytest

from ami.models.components.positional_embeddings import get_2d_positional_embeddings


@pytest.mark.parametrize("embed_dim", [384, 768])
@pytest.mark.parametrize("grid_size", [(1), (128), (1, 1), (1, 128), (128, 1), (128, 128)])
def test_get_2d_positional_embeddings(
    embed_dim: int,
    grid_size: int | tuple[int, int],
):
    positional_embeddings = get_2d_positional_embeddings(embed_dim=embed_dim, grid_size=grid_size)
    assert positional_embeddings.ndim == 2, "positional_embeddings.ndim mismatch."
    expected_n_tokens = (
        grid_size * grid_size if isinstance(grid_size, int) else grid_size[0] * grid_size[1]  # height*width
    )
    assert positional_embeddings.shape[0] == expected_n_tokens, "num of tokens mismatch."
    assert positional_embeddings.shape[1] == embed_dim, "dim mismatch."
    assert np.all(np.abs(positional_embeddings) <= 1.0), "some invalid values of sin and cos function"
