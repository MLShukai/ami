import random
from typing import Optional

import pytest
import torch

from ami.models.i_jepa import VisionTransformerEncoder, VisionTransformerPredictor


def _make_masks_randomly(
    n_mask: int,
    batch_size: int,
    n_patches_max: int,
) -> tuple[list[torch.Tensor], int]:
    """mask maker for following tests.

    Args:
        n_mask (int): Num of mask to be made.
        batch_size (int): Batch size.
        n_patches_max (int): Maximum num of patches to be selected.

    Returns:
        tuple[list[torch.Tensor], int]:
            1. Masks (len==n_mask, each shape of Tensor: [batch_size, n_patches_selected])
            2. n_patches_selected. Randomly got from the range [1, n_patches_max).
    """
    masks: list[torch.Tensor] = []
    n_patches_selected = random.randrange(n_patches_max)
    for _ in range(n_mask):
        m = []
        for _ in range(batch_size):
            m_indices, _ = torch.randperm(n_patches_max)[:n_patches_selected].sort()
            m.append(m_indices)
        masks.append(torch.stack(m, dim=0))
    return masks, n_patches_selected


class TestVisionTransformer:
    # model params
    @pytest.mark.parametrize("image_size", [224])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize(
        ["embed_dim", "depth", "num_heads", "mlp_ratio"],
        [
            [192, 12, 3, 4],  # tiny
            [384, 12, 6, 4],  # small
        ],
    )
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("n_masks_for_encoder", [None, 1, 4])
    def test_vision_transformer_encoder(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        batch_size: int,
        n_masks_for_encoder: Optional[int],
    ):
        assert image_size % patch_size == 0
        # define encoder made of ViT
        encoder = VisionTransformerEncoder(
            img_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        # define sample inputs
        images = torch.randn([batch_size, 3, image_size, image_size])
        n_patch_vertical = image_size // patch_size
        n_patch_horizontal = image_size // patch_size
        n_patches_max = n_patch_vertical * n_patch_horizontal
        # make masks for encoder
        masks_for_context_encoder = None
        if n_masks_for_encoder is not None:
            masks_for_context_encoder, n_patches_selected = _make_masks_randomly(
                n_mask=n_masks_for_encoder, batch_size=batch_size, n_patches_max=n_patches_max
            )
        # get latents
        latent = encoder(images, masks_for_context_encoder)
        # check size of output latent
        expected_batch_size = batch_size * n_masks_for_encoder if isinstance(n_masks_for_encoder, int) else batch_size
        assert latent.size(0) == expected_batch_size, "batch_size mismatch"
        expected_n_patch = (
            n_patches_selected if isinstance(n_masks_for_encoder, int) else n_patch_vertical * n_patch_horizontal
        )
        assert latent.size(1) == expected_n_patch, "num of patch mismatch"
        assert latent.size(2) == embed_dim, "embed_dim mismatch"

    # model params
    @pytest.mark.parametrize("image_size", [224])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize("context_encoder_embed_dim", [192, 384])
    @pytest.mark.parametrize("predictor_embed_dim", [384])
    @pytest.mark.parametrize("depth", [12, 16])
    @pytest.mark.parametrize("num_heads", [3, 6])
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("n_masks_for_context_encoder", [1])
    @pytest.mark.parametrize("n_masks_for_predictor", [1, 4])
    def test_vision_transformer_predictor(
        self,
        image_size: int,
        patch_size: int,
        context_encoder_embed_dim: int,
        predictor_embed_dim: int,
        depth: int,
        num_heads: int,
        batch_size: int,
        n_masks_for_context_encoder: int,
        n_masks_for_predictor: int,
    ):
        assert image_size % patch_size == 0
        n_patch_vertical = image_size // patch_size
        n_patch_horizontal = image_size // patch_size
        n_patches = n_patch_vertical * n_patch_horizontal
        # define encoder made of ViT
        predictor = VisionTransformerPredictor(
            n_patches=n_patches,
            context_encoder_embed_dim=context_encoder_embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        # define sample inputs
        n_patches_max = n_patches
        masks_for_context_encoder, n_patches_selected_for_context_encoder = _make_masks_randomly(
            n_mask=n_masks_for_context_encoder, batch_size=batch_size, n_patches_max=n_patches_max
        )
        latents = torch.randn([batch_size, n_patches_selected_for_context_encoder, context_encoder_embed_dim])
        masks_for_predictor, n_patches_selected_for_predictor = _make_masks_randomly(
            n_mask=n_masks_for_predictor, batch_size=batch_size, n_patches_max=n_patches_max
        )
        # get predictions
        predictions = predictor(
            latents=latents,
            masks_for_context_encoder=masks_for_context_encoder,
            masks_for_predictor=masks_for_predictor,
        )
        # check size of output latent
        assert predictions.size(0) == batch_size * n_masks_for_predictor, "batch_size mismatch"
        assert predictions.size(1) == n_patches_selected_for_predictor, "num of patch mismatch"
        assert predictions.size(2) == context_encoder_embed_dim, "embed_dim mismatch"
