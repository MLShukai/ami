# Ref: https://github.com/facebookresearch/ijepa

import math
import numpy as np

import torch
import torch.nn as nn

import dataclasses
from .components.vision_transformer import VisionTransformer, VisionTransformerPredictor

IJEPAEncoderSize = TypeVar(
    "IJEPAEncoderSize", Literal["tiny", "small", "base", "large", "huge", "giant"]
)


@dataclass
class IJEPAEncoderSizeParams:
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: int


encoder_sizes: Dict[IJEPAEncoderSize, IJEPAEncoderSizeParams] = {
    "tiny": IJEPAEncoderSizeParams(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4),
    "small": IJEPAEncoderSizeParams(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4),
    "base": IJEPAEncoderSizeParams(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4),
    "large": IJEPAEncoderSizeParams(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4
    ),
    "huge": IJEPAEncoderSizeParams(embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4),
    "giant": IJEPAEncoderSizeParams(
        embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48 / 11
    ),
}


class IJEPAEncoder(VisionTransformer):
    def __init__(
        self, img_size: int, patch_size: int, model_size: IJEPAEncoderSize
    ) -> None:
        """
        Args:
            img_size: Input image's size
            patch_size: How many patches to divide input image into.
            model_size: Which model size to use.
        """
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            **encoder_sizes[model_size].asdict(),
        )
        assert img_size % patch_size == 0


class IJEPAPredictor(VisionTransformerPredictor):
    def __init__(
        self,
        patch_size: int,  # encoder.patch_embed.num_patches,
        embed_dim: int,  # encoder.embed_dim,
        num_heads: int,  # encoder.num_heads
        predictor_embed_dim: int = 384,
        depth: int = 12,
    ) -> None:
        """
        Args:
            patch_size: How many patches to divide input image into.
            embed_dim: Embed dim of I-JEPA's context encoder.
            num_heads: num_heads of I-JEPA's context encoder.
            predictor_embed_dim: Output tensor's dimension.
            depth: Num of transformers in this predictor.
        """
        super().__init__(
            patch_size=patch_size,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=pred_depth,
            num_heads=num_heads,
        )
