import random

import numpy as np
import pytest

from ami.models.components.positional_embeddings import get_2d_positional_embeddings


@pytest.mark.parametrize("embed_dim", [384, 768])
@pytest.mark.parametrize(
    "grid_size",
    [
        # Check whether to pass when even with a single value.
        (1),
        (128),
        # Check whether to pass when either one is 1.
        (1, 128),
        (128, 1),
        # Check whether to pass when two values are different and not both 1's.
        (128, 64),
        (64, 128),
        # Check whether to pass when two values are the same number.
        (1, 1),
        (128, 128),
    ],
)
def test_get_2d_positional_embeddings(
    embed_dim: int,
    grid_size: int | tuple[int, int],
):
    positional_embeddings = get_2d_positional_embeddings(embed_dim=embed_dim, grid_size=grid_size)
    assert positional_embeddings.ndim == 3, "positional_embeddings.ndim mismatch."
    (expected_grid_size_h, expected_grid_size_w) = (
        (grid_size, grid_size) if isinstance(grid_size, int) else (grid_size[0], grid_size[1])
    )
    assert positional_embeddings.shape[0] == expected_grid_size_h, "grid size (height) mismatch."
    assert positional_embeddings.shape[1] == expected_grid_size_w, "grid size (width) mismatch."
    assert positional_embeddings.shape[2] == embed_dim, "dim mismatch."
    assert np.all(np.abs(positional_embeddings) <= 1.0), "some invalid values of sin and cos function"
