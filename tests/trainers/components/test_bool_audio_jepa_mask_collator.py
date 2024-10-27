import pytest
import torch

from ami.trainers.components.bool_audio_jepa_mask_collator import (
    BoolAudioJEPAMultiBlockMaskCollator,
)


class TestBoolAudioJEPAMultiBlockMaskCollator:
    # @pytest.mark.parametrize("audio_size", [224])
    # @pytest.mark.parametrize("patch_size", [16])
    # @pytest.mark.parametrize("min_keep", [10])
    # @pytest.mark.parametrize("mask_scale", [(0.1, 0.25)])
    # def test_sample_mask(self, audio_size, patch_size, min_keep, mask_scale):
    #     collator = BoolAudioJEPAMultiBlockMaskCollator(
    #         input_size=input_size:,
    #         patch_size=patch_size,
    #         stride=stride
    #         mask_scale=mask_scale,
    #         n_masks: int = 4,
    #         min_keep: int = 10,
    #     )
    #     g = torch.Generator()
    #     n_patches = (audio_size // patch_size) ** 2
    #     for _ in range(100):
    #         top, bottom, left, right = collator._sample_mask_rectangle(g)
    #         assert top < bottom
    #         assert top >= 0
    #         assert bottom <= audio_size
    #         assert left < right
    #         assert left >= 0
    #         assert bottom <= audio_size

    #         height, width = (bottom - top), (right - left)
    #         # test mask scale
    #         mask_scale_min, mask_scale_max = mask_scale
    #         assert height * width <= mask_scale_max * n_patches
    #         assert height * width >= mask_scale_min * n_patches
    #         # test min keep
    #         assert height * width >= min_keep

    # collator params
    @pytest.mark.parametrize(
        ["audio_size", "patch_size", "stride", "n_masks", "min_keep"],
        [
            [400, 400, 320, 4, 0], 
            [16080, 400, 320, 4, 10], 
        ],
    )
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("n_channels", [1, 2]) # monoral and stereo audio respectively
    def test_bool_i_jepa_mask_collator(
        self,
        audio_size: int,
        patch_size: int,
        stride: int,
        n_masks: int,
        min_keep: int,
        batch_size: int,
        n_channels: int,
    ):
        assert patch_size<=audio_size
        assert stride<=audio_size
        assert (audio_size - (patch_size - stride))%stride == 0
        # define BoolAudioJEPAMultiBlockMaskCollator
        collator = BoolAudioJEPAMultiBlockMaskCollator(
            input_size=audio_size,
            patch_size=patch_size,
            stride=stride,
            n_masks=n_masks,
            min_keep=min_keep,
        )
        # define sample inputs
        audios = [(torch.randn([n_channels, audio_size]),) for _ in range(batch_size)]
        # collate batch and create masks
        (
            collated_audios,
            collated_encoder_masks,
            collated_predictor_targets,
        ) = collator(audios)

        # check image sizes
        assert collated_audios.size(0) == batch_size, "batch_size mismatch"
        assert collated_audios.size(1) == n_channels, "channels mismatch"
        assert collated_audios.size(2) == audio_size, "collated_audios num of samples mismatch"

        # calc num of patches
        n_patches = (audio_size - (patch_size - stride)) / stride

        # check masks for context encoder
        assert collated_encoder_masks.dim() == 2
        assert collated_encoder_masks.size(0) == batch_size, "batch_size mismatch (collated_encoder_masks)"
        assert collated_encoder_masks.size(1) == n_patches, "patch count mismatch (collated_encoder_masks)"
        assert collated_encoder_masks.dtype == torch.bool, "dtype mismatch (collated_encoder_masks)"

        # check masks for predictor target
        assert collated_predictor_targets.dim() == 2
        assert collated_predictor_targets.size(0) == batch_size, "batch_size mismatch (collated_predictor_targets)"
        assert collated_predictor_targets.size(1) == n_patches, "patch count mismatch (collated_predictor_targets)"
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

    # def test_sample_masks_and_target(self):
    #     audio_size, patch_size = 224, 16
    #     collator = BoolAudioJEPAMultiBlockMaskCollator(
    #         input_size=(audio_size, audio_size),
    #         patch_size=(patch_size, patch_size),
    #         n_masks=4,
    #         min_keep=50,
    #     )
    #     g = torch.Generator()
    #     encoder_mask, predictor_target = collator.sample_masks_and_target(g)

    #     n_patches = (audio_size // patch_size) ** 2

    #     assert encoder_mask.shape == (n_patches,)
    #     assert predictor_target.shape == (n_patches,)
    #     assert encoder_mask.dtype == torch.bool
    #     assert predictor_target.dtype == torch.bool

    #     # Check that at least min_keep patches are unmasked for encoder
    #     assert torch.sum(~encoder_mask) >= collator.min_keep

    #     # Check that at least one patch is masked for predictor target
    #     assert torch.sum(predictor_target) > 0

    #     # Check that encoder mask and predictor target are not identical
    #     assert not torch.all(encoder_mask == predictor_target)
