import pytest
import torch

from ami.trainers.components.i_jepa_mask_collator import (
    IJEPAMaskCollator
)


class TestIJEPAMaskCollator:

    # collator params
    @pytest.mark.parametrize("image_size", [224, 512])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize(
        ["n_masks_for_context_encoder", "n_masks_for_predictor"],
        [
            [1, 4],
        ],
    )
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_i_jepa_mask_collator(
        self,
        image_size: int,
        patch_size: int,
        n_masks_for_context_encoder: int,
        n_masks_for_predictor: int,
        batch_size: int,
    ):
        assert image_size % patch_size == 0
        # define IJEPAMaskCollator
        collator = IJEPAMaskCollator(
            input_size=(image_size, image_size),
            patch_size=patch_size,
            enc_mask_scale=(0.85, 1.0),
            pred_mask_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5),
            n_masks_for_context_encoder=n_masks_for_context_encoder,
            n_masks_for_predictor=n_masks_for_predictor,
            min_keep=10,
            allow_overlap=False,
        )
        # define sample inputs
        images = [torch.randn([3, image_size, image_size]) for _ in range(batch_size)]
        # collate batch and create masks.
        (
            collated_images,
            collated_masks_for_context_encoder,
            collated_masks_for_predictor,
        ) = collator(images)
        # check image sizes
        assert collated_images.size(0)==batch_size, "batch_size mismatch"
        assert collated_images.size(1)==3, "batch_size mismatch"
        assert collated_images.size(2)==image_size, "collated_images height mismatch"
        assert collated_images.size(3)==image_size, "collated_images width mismatch"
        # calc num of patch
        n_patch_vertical = image_size // patch_size
        n_patch_horizontal = image_size // patch_size
        n_patch = n_patch_vertical * n_patch_horizontal
        # check about masks for context encoder
        assert len(collated_masks_for_context_encoder)==n_masks_for_context_encoder
        for masks_for_context_encoder in collated_masks_for_context_encoder:
            assert masks_for_context_encoder.dim()==2
            assert masks_for_context_encoder.size(0)==batch_size, "batch_size mismatch (masks_for_context_encoder)"
            assert masks_for_context_encoder.size(1)<=n_patch, "too much size (masks_for_context_encoder)"
            assert torch.all(0<=masks_for_context_encoder) and torch.all(masks_for_context_encoder<n_patch), "invalid indices (masks_for_context_encoder)"
        # check about masks for predictor
        assert len(collated_masks_for_predictor)==n_masks_for_predictor
        for masks_for_predictor in collated_masks_for_predictor:
            assert masks_for_predictor.dim()==2
            assert masks_for_predictor.size(0)==batch_size, "batch_size mismatch (masks_for_predictor)"
            assert masks_for_predictor.size(1)<=n_patch, "too much size (masks_for_predictor)"
            assert torch.all(0<=masks_for_predictor) and torch.all(masks_for_predictor<n_patch), "invalid indices (masks_for_predictor)"
