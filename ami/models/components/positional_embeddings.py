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
            positional embeddings (shape: [grid_size[0]*grid_size[1], embed_dim]).
    """
    grid_size_h, grid_size_w = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
    grid_h = np.arange(grid_size_h, dtype=float)
    grid_w = np.arange(grid_size_w, dtype=float)
    meshgrid = np.meshgrid(grid_w, grid_h)  # here w goes first as args
    grid = np.stack(meshgrid, axis=0)  # [2, grid_size_h, grid_size_w]

    grid = np.expand_dims(grid, 1)  # [2, 1, grid_size_h, grid_size_w]
    positional_embeddings = get_2d_sincos_positional_embeddings_from_grid(embed_dim, grid)
    return positional_embeddings


def get_2d_sincos_positional_embeddings_from_grid(
    embed_dim: int, grid: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    embeddings_h = get_1d_sincos_positional_embeddings_from_grid(
        embed_dim // 2, grid[0]
    )  # [grid_size_h*grid_size_w, embed_dim//2]
    embeddings_w = get_1d_sincos_positional_embeddings_from_grid(
        embed_dim // 2, grid[1]
    )  # [grid_size_h*grid_size_w, embed_dim//2]

    embeddings = np.concatenate([embeddings_h, embeddings_w], axis=1)  # [grid_size_h*grid_size_w, embed_dim]
    return embeddings


def get_1d_sincos_positional_embeddings_from_grid(
    embed_dim: int, positions: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Args:
        embed_dim (int): dim of positional embeddings.
        positions (npt.NDArray[np.float64]): positions to be encoded (shape: [1, grid_size_h, grid_size_w]).
    Returns:
        npt.NDArray[np.float64]:
            positional embeddings (shape: [grid_size_h*grid_size_w, embed_dim]).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # [embed_dim//2]

    positions = positions.reshape(-1)  # [grid_size_h*grid_size_w]
    outer = np.outer(positions, omega)  # [grid_size_h*grid_size_w, embed_dim//2]

    positional_embeddings_sin = np.sin(outer)  # [grid_size_h*grid_size_w, embed_dim//2]
    positional_embeddings_cos = np.cos(outer)  # [grid_size_h*grid_size_w, embed_dim//2]

    positional_embeddings = np.concatenate(
        [positional_embeddings_sin, positional_embeddings_cos], axis=1
    )  # [grid_size_h*grid_size_w, embed_dim]
    return positional_embeddings
