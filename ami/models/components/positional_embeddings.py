# Ref: https://github.com/facebookresearch/ijepa

import numpy as np
import numpy.typing as npt


def get_2d_positional_embeddings(embed_dim: int, grid_size: int | tuple[int, int]) -> npt.NDArray[np.float64]:
    """
    Args:
        embed_dim (int): dim of positional embeddings.
        grid_size (int | tuple[int,int]): int of the grid height and width.
    Returns:
        npt.NDArray[np.float64]:
            positional embeddings (shape: [grid_size_h, grid_size_w, embed_dim]).
    """
    grid_size_h, grid_size_w = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
    grid_h = np.arange(grid_size_h, dtype=float)
    grid_w = np.arange(grid_size_w, dtype=float)
    meshgrid = np.meshgrid(grid_w, grid_h)  # here w goes first as args
    grid = np.stack(meshgrid, axis=0)  # [2, grid_size_h, grid_size_w]

    positional_embeddings = get_2d_sincos_positional_embeddings_from_grid(embed_dim, grid)
    return positional_embeddings


def get_1d_positional_embeddings(embed_dim: int, sequence_length: int) -> npt.NDArray[np.float64]:
    """
    Args:
        embed_dim (int): dim of positional embeddings.
        sequence_length (int): length of positional embeddings.
    Returns:
        npt.NDArray[np.float64]:
            positional embeddings (shape: [sequence_length, embed_dim]).
    """
    positional_embeddings = get_2d_positional_embeddings(
        embed_dim=embed_dim, grid_size=(1, sequence_length)
    )  # [1, sequence_length, embed_dim]
    positional_embeddings = np.squeeze(positional_embeddings, axis=0)  # [sequence_length, embed_dim]
    return positional_embeddings


def get_2d_sincos_positional_embeddings_from_grid(
    embed_dim: int, grid: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Args:
        embed_dim (int): dim of positional embeddings.
        grid (npt.NDArray[np.float64]): positions to be encoded is represented as grid(shape: [2, grid_size_h, grid_size_w]).
    Returns:
        npt.NDArray[np.float64]:
            positional embeddings (shape: [grid_size_h, grid_size_w, embed_dim]).
    """

    assert embed_dim % 2 == 0
    assert grid.shape[0] == 2  # grid_h, grid_w

    # use half of dimensions to encode grid_h
    embeddings_h = get_1d_sincos_positional_embeddings(
        embed_dim // 2, grid[0].reshape(-1)
    )  # [grid_size_h*grid_size_w, embed_dim//2]
    embeddings_w = get_1d_sincos_positional_embeddings(
        embed_dim // 2, grid[1].reshape(-1)
    )  # [grid_size_h*grid_size_w, embed_dim//2]

    embeddings = np.concatenate([embeddings_h, embeddings_w], axis=-1)  # [grid_size_h*grid_size_w, embed_dim]
    _, grid_size_h, grid_size_w = grid.shape
    return embeddings.reshape(grid_size_h, grid_size_w, embed_dim)


def get_1d_sincos_positional_embeddings(embed_dim: int, positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Args:
        embed_dim (int): dim of positional embeddings.
        positions (npt.NDArray[np.float64]): positions to be encoded (shape: [length, ]).
    Returns:
        npt.NDArray[np.float64]:
            positional embeddings (shape: [length, embed_dim]).
    """
    assert embed_dim % 2 == 0
    assert positions.ndim == 1
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # [embed_dim//2]
    outer = np.outer(positions, omega)  # [length, embed_dim//2]

    positional_embeddings_sin = np.sin(outer)  # [length, embed_dim//2]
    positional_embeddings_cos = np.cos(outer)  # [length, embed_dim//2]

    positional_embeddings = np.concatenate(
        [positional_embeddings_sin, positional_embeddings_cos], axis=-1
    )  # [length, embed_dim]
    return positional_embeddings
