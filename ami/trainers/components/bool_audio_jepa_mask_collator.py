import math
import random
from multiprocessing import Value

import torch
from torch import Tensor
from torch.utils.data import default_collate

from ami.models.utils import size_2d, size_2d_to_int_tuple


class BoolAudioJEPAMultiBlockMaskCollator:
    """Audio-JEPA collator function for providing boolean mask tensors.

    This collator creates boolean masks for both the context encoder and predictor target.
    It's designed to work with the Audio-JEPA (Joint Embedding Predictive Architecture) model.

    The masks are boolean tensors where:
    - True values indicate patches to be masked (ignored)
    - False values indicate patches to be processed or predicted
    """

    def __init__(
        self,
        input_sample_size: int,
        patch_sample_size: int,
        stride: int,
        mask_scale: tuple[float, float] = (0.10, 0.25),  # masking 1/10 ~ 1/4 region
        n_masks: int = 4,
        min_keep: int = 10,
    ) -> None:
        """Initialize the BoolAudioJEPAMultiBlockMaskCollator.

        Args:
            input_sample_size (size_2d):
                Num of samples in the input audio.
            patch_sample_size (int):
                Size of each patch. It can also be regarded as window_size.
            stride (int):
                Stride during making patches from audio. It can also be regarded as hop_size.
            mask_scale (tuple[float, float]):
                Range of mask scale (min, max).
            n_masks (int):
                Number of mask candidates to generate.
            min_keep (int):
                Minimum number of patches to keep unmasked.
        """
        super().__init__()
        assert mask_scale[0] < mask_scale[1]
        assert mask_scale[0] > 0
        assert mask_scale[1] < 1

        assert patch_sample_size <= input_sample_size
        assert stride <= patch_sample_size
        assert (input_sample_size - (patch_sample_size - stride)) % stride == 0
        self.input_sample_size = input_sample_size
        self.patch_sample_size = patch_sample_size

        self.n_patches = (input_sample_size - (self.patch_sample_size - stride)) // stride
        assert min_keep <= self.n_patches

        self.mask_scale = mask_scale
        self.n_masks = n_masks
        self.min_keep = min_keep  # minimum number of patches to keep unmasked
        self._itr_counter = Value("i", random.randrange(2**32))  # collator is shared across worker processes

    def step(self) -> int:
        """Increment and return the iteration counter."""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_mask(
        self,
        generator: torch.Generator,
    ) -> tuple[int, int]:
        """Randomly sample a mask on patches.

        Args:
            generator (torch.Generator): Generator for pseudo-random numbers.

        Returns:
            tuple[int, int]: Start, and end coordinates of the mask.
        """
        # Sample mask scale
        min_s, max_s = self.mask_scale
        mask_scale = min_s + torch.rand(1, generator=generator).item() * (max_s - min_s)

        # Given scale, compute n_samples of mask.
        num_patches_max = self.n_patches
        mask_sample_size = round(mask_scale * num_patches_max)

        # clamp with [1, self.n_patches]
        mask_sample_size = min(max(mask_sample_size, 1), self.n_patches)

        # Compute mask coordinates
        start = int(torch.randint(high=self.n_patches - mask_sample_size + 1, size=(1,), generator=generator).item())
        end = start + mask_sample_size
        return start, end

    def sample_masks_and_target(self, generator: torch.Generator) -> tuple[Tensor, Tensor]:
        """Sample boolean masks for the encoder and a target mask for the
        predictor.

        Args:
            generator (torch.Generator):
                Generator for pseudo-random numbers.

        Returns:
            tuple[Tensor, Tensor]:
                - encoder_mask:
                    Boolean mask for the encoder (True for masked patches)
                - predictor_target:
                    Boolean mask representing the target for the predictor (True for masked patches)
        """
        sampled_masks = []
        for _ in range(self.n_masks):
            mask = torch.zeros(self.n_patches, dtype=torch.bool)
            start, end = self._sample_mask(generator)
            mask[start:end] = True
            sampled_masks.append(mask.flatten())

        # Create encoder mask by combining all sampled masks
        encoder_mask = torch.stack(sampled_masks).sum(0).type(torch.bool)
        # Randomly select one mask as the predictor target
        predictor_target = sampled_masks[
            int(torch.randint(high=len(sampled_masks), size=(1,), generator=generator).item())
        ]

        # Apply min keep
        if encoder_mask.logical_not().sum() < self.min_keep:
            indices = torch.randperm(self.n_patches, generator=generator)[: self.min_keep]
            encoder_mask[indices] = False
            predictor_target[indices] = True

        return encoder_mask, predictor_target

    def __call__(self, audios: list[tuple[Tensor]]) -> tuple[Tensor, Tensor, Tensor]:
        """Collate input audios and create boolean masks for context encoder
        and predictor target.

        Args:
            audios (list[tuple[Tensor]]):
                List of audio tensors. Each audio is shape [n_channels, n_samples].

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                - collated_audios:
                    Collated audios (shape: [batch_size, n_channels, n_samples])
                - collated_encoder_masks:
                    Boolean masks for context encoder (shape: [batch_size, n_patches])
                - collated_predictor_targets:
                    Boolean masks representing predictor targets (shape: [batch_size, n_patches])
        """
        collated_audios: Tensor = default_collate(audios)[0]
        assert collated_audios.size(-1) == self.input_sample_size

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        encoder_masks, predictor_targets = [], []
        for _ in range(len(audios)):
            enc_mask, pred_target = self.sample_masks_and_target(g)
            encoder_masks.append(enc_mask)
            predictor_targets.append(pred_target)

        collated_encoder_masks = torch.stack(encoder_masks)
        collated_predictor_targets = torch.stack(predictor_targets)

        return (
            collated_audios,
            collated_encoder_masks,
            collated_predictor_targets,
        )
