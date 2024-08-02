import random

import pytest
import torch

from ami.models.components.vision_transformer_layer import VisionTransformerLayer


class TestVisionTransformerLayer:
    # model params
    @pytest.mark.parametrize("embedding_dim", [384, 768])
    @pytest.mark.parametrize("num_heads", [3, 6])
    @pytest.mark.parametrize("mlp_ratio", [4.0, 48/11])
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("max_n_patches", [1, 512])
    def test_vision_transformer_layer(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: float,
        batch_size: int,
        max_n_patches: int,
    ):
        vision_transformer_layer = VisionTransformerLayer(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        # define input
        n_patches = random.randint(1, max_n_patches) # including max_n_patches
        input_patches = torch.randn([batch_size, n_patches, embedding_dim], dtype=torch.float)
        output_patches = vision_transformer_layer(input_patches)
        assert input_patches.size()==output_patches.size()
