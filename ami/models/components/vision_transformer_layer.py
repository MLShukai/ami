# Ref: https://github.com/facebookresearch/ijepa

import math

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from .drop_path import DropPath


class MLP(nn.Module):
    """Multi Layer Perceptron."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Attention Layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if qk_scale is None:
            qk_scale = head_dim**-0.5
        assert qk_scale != 0.0
        self.scale = qk_scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class VisionTransformerLayer(nn.Module):
    """Vision Transformer Layer."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.attn = Attention(
            embedding_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embedding_dim, eps=1e-6)
        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embedding_dim,
            hidden_features=mlp_hidden_dim,
            out_features=embedding_dim,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Vision Transformer.

        Args:
            x (torch.Tensor): Shape is [batch_size, n_patches, embedding_dim]

        Returns:
            torch.Tensor: Shape is same as input.
        """
        y, _ = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
