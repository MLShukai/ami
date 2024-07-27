import pytest
import torch

from ami.models.components.vision_transformer import (
    VisionTransformer,
    VisionTransformerPredictor,
)


class TestVisionTransformer:

    # model params
    @pytest.mark.parametrize("image_size", [224, 1024, 1920])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize(
        ["embed_dim", "depth", "num_heads", "mlp_ratio"],
        [
            [192, 12, 3, 4],  # tiny
            [384, 12, 6, 4],  # small
            [768, 12, 12, 4],  # base
            [1024, 24, 16, 4],  # large
            [1280, 32, 16, 4],  # huge
            [1408, 40, 16, 48 / 11],  # giant
        ],
    )
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 32])
    @pytest.mark.parametrize("use_masks", [True, False])
    def test_vision_transformer(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        batch_size: int,
        use_masks: bool,
    ):
        assert image_size % patch_size == 0
        vit = VisionTransformer(
            img_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        images = torch.randn([batch_size, image_size, image_size])
        n_patch = image_size // patch_size
        masks_for_context_encoder = (
            torch.randn([batch_size, 1, n_patch * n_patch]) if use_masks else None
        )
        latent = vits(images, masks_for_context_encoder)
