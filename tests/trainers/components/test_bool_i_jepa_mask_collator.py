import pytest
import torch

from ami.trainers.components.bool_i_jepa_mask_collator import (
    BoolIJEPAMultiBlockMaskCollator,
)


class TestBoolIJEPAMultiBlockMaskCollator:
    @pytest.mark.parametrize("image_size", [224])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize("min_keep", [10])
    @pytest.mark.parametrize("mask_scale", [(0.1, 0.25)])
    def test_sample_mask_rectangle(self, image_size, patch_size, min_keep, mask_scale):
        collator = BoolIJEPAMultiBlockMaskCollator(
            input_size=image_size, patch_size=patch_size, mask_scale=mask_scale, min_keep=min_keep
        )
        g = torch.Generator()
        n_patches = (image_size // patch_size) ** 2
        for _ in range(100):
            top, bottom, left, right = collator._sample_mask_rectangle(g)
            assert top < bottom
            assert top >= 0
            assert bottom <= image_size
            assert left < right
            assert left >= 0
            assert bottom <= image_size

            height, width = (bottom - top), (right - left)
            # test mask scale
            mask_scale_min, mask_scale_max = mask_scale
            assert height * width <= mask_scale_max * n_patches
            assert height * width >= mask_scale_min * n_patches
            # test min keep
            assert height * width >= min_keep

    # collator params
    @pytest.mark.parametrize("image_size", [224, 512])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize("n_masks", [4])
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_bool_i_jepa_mask_collator(
        self,
        image_size: int,
        patch_size: int,
        n_masks: int,
        batch_size: int,
    ):
        assert image_size % patch_size == 0
        # define BoolIJEPAMultiBlockMaskCollator
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
            collated_encoder_masks,
            collated_predictor_targets,
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
        assert collated_encoder_masks.dim() == 2
        assert collated_encoder_masks.size(0) == batch_size, "batch_size mismatch (collated_encoder_masks)"
        assert collated_encoder_masks.size(1) == n_patch, "patch count mismatch (collated_encoder_masks)"
        assert collated_encoder_masks.dtype == torch.bool, "dtype mismatch (collated_encoder_masks)"

        # check masks for predictor target
        assert collated_predictor_targets.dim() == 2
        assert collated_predictor_targets.size(0) == batch_size, "batch_size mismatch (collated_predictor_targets)"
        assert collated_predictor_targets.size(1) == n_patch, "patch count mismatch (collated_predictor_targets)"
        assert collated_predictor_targets.dtype == torch.bool, "dtype mismatch (collated_predictor_targets)"

        # check that at least min_keep patches are unmasked for encoder
        assert (
            torch.sum(~collated_encoder_masks, dim=1).min() >= collator.min_keep
        ), "min_keep not satisfied for encoder"

        # check that at least one patch is masked for predictor target
        assert torch.sum(collated_predictor_targets, dim=1).min() > 0, "no prediction target for predictor"

        # check that encoder masks and predictor targets are not identical
        assert not torch.all(
            collated_encoder_masks == collated_predictor_targets
        ), "encoder masks and predictor targets must be different"

    def test_sample_masks_and_target(self):
        image_size, patch_size = 224, 16
        collator = BoolIJEPAMultiBlockMaskCollator(
            input_size=(image_size, image_size),
            patch_size=(patch_size, patch_size),
            n_masks=4,
            min_keep=50,
        )
        g = torch.Generator()
        encoder_mask, predictor_target = collator.sample_masks_and_target(g)

        n_patches = (image_size // patch_size) ** 2

        assert encoder_mask.shape == (n_patches,)
        assert predictor_target.shape == (n_patches,)
        assert encoder_mask.dtype == torch.bool
        assert predictor_target.dtype == torch.bool

        # Check that at least min_keep patches are unmasked for encoder
        assert torch.sum(~encoder_mask) >= collator.min_keep

        # Check that at least one patch is masked for predictor target
        assert torch.sum(predictor_target) > 0

        # Check that encoder mask and predictor target are not identical
        assert not torch.all(encoder_mask == predictor_target)
