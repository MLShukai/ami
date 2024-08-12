import pytest
import torch

from ami.trainers.components.bool_i_jepa_mask_collator import (
    BoolIJEPAMultiBlockMaskCollator,
)


class TestBoolIJEPAMultiBlockMaskCollator:

    # collator params
    @pytest.mark.parametrize("image_size", [224, 512])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize("n_masks", [4])
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_i_jepa_bool_mask_collator(
        self,
        image_size: int,
        patch_size: int,
        n_masks: int,
        batch_size: int,
    ):
        assert image_size % patch_size == 0
        # define IJEPABoolMaskCollator
        collator = BoolIJEPAMultiBlockMaskCollator(
            input_size=(image_size, image_size),
            patch_size=(patch_size, patch_size),
            n_masks=n_masks,
            min_keep=50,
        )
        # define sample inputs
        images = [(torch.randn([3, image_size, image_size]),) for _ in range(batch_size)]
        # collate batch and create masks
        (
            collated_images,
            collated_masks_for_context_encoder,
            collated_masks_for_predictor,
        ) = collator(images)

        # check image sizes
        assert collated_images.size(0) == batch_size, "batch_size mismatch"
        assert collated_images.size(1) == 3, "channel mismatch"
        assert collated_images.size(2) == image_size, "collated_images height mismatch"
        assert collated_images.size(3) == image_size, "collated_images width mismatch"

        # calc num of patches
        n_patch_vertical = image_size // patch_size
        n_patch_horizontal = image_size // patch_size
        n_patch = n_patch_vertical * n_patch_horizontal

        # check masks for context encoder
        assert collated_masks_for_context_encoder.dim() == 2
        assert (
            collated_masks_for_context_encoder.size(0) == batch_size
        ), "batch_size mismatch (masks_for_context_encoder)"
        assert collated_masks_for_context_encoder.size(1) == n_patch, "patch count mismatch (masks_for_context_encoder)"
        assert collated_masks_for_context_encoder.dtype == torch.bool, "dtype mismatch (masks_for_context_encoder)"

        # check masks for predictor
        assert collated_masks_for_predictor.dim() == 2
        assert collated_masks_for_predictor.size(0) == batch_size, "batch_size mismatch (masks_for_predictor)"
        assert collated_masks_for_predictor.size(1) == n_patch, "patch count mismatch (masks_for_predictor)"
        assert collated_masks_for_predictor.dtype == torch.bool, "dtype mismatch (masks_for_predictor)"

        # check that at least min_keep patches are unmasked for encoder
        assert (
            torch.sum(~collated_masks_for_context_encoder, dim=1).min() >= collator.min_keep
        ), "min_keep not satisfied for encoder"

        # check that at least one patch is unmasked for predictor
        assert torch.sum(~collated_masks_for_predictor, dim=1).min() > 0, "no prediction target for predictor"

        # check that encoder and predictor masks are not identical
        assert not torch.all(
            collated_masks_for_context_encoder == collated_masks_for_predictor
        ), "encoder and predictor masks are identical"
