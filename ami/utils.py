from enum import Enum

import torch


def min_max_normalize(
    x: torch.Tensor, new_min: float = 0.0, new_max: float = 1.0, eps: float = 1e-8, dim: int | None = None
) -> torch.Tensor:
    """Perform min-max normalization on a PyTorch tensor.

    Args:
        x (torch.Tensor): Input tensor to normalize
        new_min (float): Minimum value of the new range (default: 0.0)
        new_max (float): Maximum value of the new range (default: 1.0)
        eps (float): Small value to avoid division by zero (default: 1e-8)
        dim (int | None): Dimension along which to normalize. If None, normalize over all dimensions.

    Returns:
        torch.Tensor: Normalized tensor
    """
    if dim is None:
        x_min = x.min()
        x_max = x.max()
    else:
        x_min = x.min(dim=dim, keepdim=True).values
        x_max = x.max(dim=dim, keepdim=True).values

    denominator = (x_max - x_min).clamp(min=eps)

    return (x - x_min) / denominator * (new_max - new_min) + new_min


class Modality(str, Enum):
    """Enumerates modalities."""

    IMAGE = "image"
    AUDIO = "audio"
