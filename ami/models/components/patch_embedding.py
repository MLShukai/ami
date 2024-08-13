import torch
import torch.nn as nn

from ..utils import size_2d


class PatchEmbedding(nn.Module):
    """Convert input images into patch embeddings."""

    def __init__(
        self,
        patch_size: size_2d = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """

        Args:
            patch_size (size_2d):
                Pixel size per a patch.
                Defaults to 16.
            in_channels (int):
                Num of input images channels.
                Defaults to 3.
            embed_dim (int):
                Num of embed dimensions per a patch
                Defaults to 768.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    # (batch, channels, height, width) -> (batch, n_patches, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(-2).transpose(-2, -1)
        return x
