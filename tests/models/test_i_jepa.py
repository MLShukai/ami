import random
from typing import Optional

import pytest
import torch

from ami.models.i_jepa import IJEPAEncoder, IJEPAPredictor


def _make_patch_selections_randomly(
    n_patch_selections: int,
    batch_size: int,
    n_patches_max: int,
) -> tuple[list[torch.Tensor], int]:
    """Patch selections maker for following tests.

    Args:
        n_patch_selections (int): Num of patch_selections to be made.
        batch_size (int): Batch size.
        n_patches_max (int): Maximum num of patches to be selected.

    Returns:
        tuple[list[torch.Tensor], int]:
            1. patch_selections (len==n_patch_selections, each shape of Tensor: [batch_size, n_patches_selected])
               Each patch_selection contains indices of patches to be selected.
            2. n_patches_selected. Randomly got from the range [1, n_patches_max).
    """
    patch_selections: list[torch.Tensor] = []
    n_patches_selected = random.randrange(n_patches_max)
    for _ in range(n_patch_selections):
        m = []
        for _ in range(batch_size):
            m_indices, _ = torch.randperm(n_patches_max)[:n_patches_selected].sort()
            m.append(m_indices)
        patch_selections.append(torch.stack(m, dim=0))
    return patch_selections, n_patches_selected


class TestVisionTransformer:
    # model params
    @pytest.mark.parametrize("image_size", [224])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize(
        ["embed_dim", "depth", "num_heads", "mlp_ratio"],
        [
            [8, 2, 2, 4],  # tiny
            [32, 2, 4, 4],  # small
        ],
    )
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("n_patch_selections_for_encoder", [None, 1, 4])
    def test_vision_transformer_encoder(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        batch_size: int,
        n_patch_selections_for_encoder: Optional[int],
    ):
        assert image_size % patch_size == 0
        # define encoder made of ViT
        encoder = IJEPAEncoder(
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
        # make patch_selections for encoder
        patch_selections_for_context_encoder = None
        if n_patch_selections_for_encoder is not None:
            patch_selections_for_context_encoder, n_patches_selected = _make_patch_selections_randomly(
                n_patch_selections=n_patch_selections_for_encoder, batch_size=batch_size, n_patches_max=n_patches_max
            )
        # get latents
        latent = encoder(images=images, patch_selections_for_context_encoder=patch_selections_for_context_encoder)
        # check size of output latent
        expected_batch_size = (
            batch_size * n_patch_selections_for_encoder
            if isinstance(n_patch_selections_for_encoder, int)
            else batch_size
        )
        assert latent.size(0) == expected_batch_size, "batch_size mismatch"
        expected_n_patch = (
            n_patches_selected
            if isinstance(n_patch_selections_for_encoder, int)
            else n_patch_vertical * n_patch_horizontal
        )
        assert latent.size(1) == expected_n_patch, "num of patch mismatch"
        assert latent.size(2) == embed_dim, "embed_dim mismatch"

    # model params
    @pytest.mark.parametrize("image_size", [224])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize("context_encoder_embed_dim", [8])
    @pytest.mark.parametrize("predictor_embed_dim", [32])
    @pytest.mark.parametrize("depth", [2])
    @pytest.mark.parametrize("num_heads", [2, 4])
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize(
        ["n_patch_selections_for_context_encoder", "n_patch_selections_for_predictor"],
        [
            # Check whether to pass when two values are the same number.
            [1, 1],
            [4, 4],
            # Check whether to pass when two values are different.
            [1, 4],  # same setting as the original paper.
            [4, 1],
        ],
    )
    def test_vision_transformer_predictor(
        self,
        image_size: int,
        patch_size: int,
        context_encoder_embed_dim: int,
        predictor_embed_dim: int,
        depth: int,
        num_heads: int,
        batch_size: int,
        n_patch_selections_for_context_encoder: int,
        n_patch_selections_for_predictor: int,
    ):
        assert image_size % patch_size == 0
        n_patch_vertical = image_size // patch_size
        n_patch_horizontal = image_size // patch_size
        n_patches = n_patch_vertical * n_patch_horizontal
        # define encoder made of ViT
        predictor = IJEPAPredictor(
            n_patches=n_patches,
            context_encoder_embed_dim=context_encoder_embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        # define sample inputs
        n_patches_max = n_patches
        patch_selections_for_context_encoder, n_patches_selected_for_context_encoder = _make_patch_selections_randomly(
            n_patch_selections=n_patch_selections_for_context_encoder,
            batch_size=batch_size,
            n_patches_max=n_patches_max,
        )
        latents = torch.randn(
            [
                batch_size * n_patch_selections_for_context_encoder,
                n_patches_selected_for_context_encoder,
                context_encoder_embed_dim,
            ]
        )
        patch_selections_for_predictor, n_patches_selected_for_predictor = _make_patch_selections_randomly(
            n_patch_selections=n_patch_selections_for_predictor, batch_size=batch_size, n_patches_max=n_patches_max
        )
        # get predictions
        predictions = predictor(
            latents=latents,
            patch_selections_for_context_encoder=patch_selections_for_context_encoder,
            patch_selections_for_predictor=patch_selections_for_predictor,
        )
        # check size of output latent
        assert (
            predictions.size(0)
            == batch_size * n_patch_selections_for_context_encoder * n_patch_selections_for_predictor
        ), "batch_size mismatch"
        assert predictions.size(1) == n_patches_selected_for_predictor, "num of patch mismatch"
        assert predictions.size(2) == context_encoder_embed_dim, "embed_dim mismatch"
