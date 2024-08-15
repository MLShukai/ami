import math
import random
from multiprocessing import Value

import torch
from torch import Tensor
from torch.utils.data import default_collate

from ami.models.utils import size_2d, size_2d_to_int_tuple


class BoolIJEPAMultiBlockMaskCollator:
    """I-JEPA collator function for providing boolean mask tensors.

    This collator creates boolean masks for both the context encoder and predictor target.
    It's designed to work with the I-JEPA (Image Joint Embedding Predictive Architecture) model.

    The masks are boolean tensors where:
    - True values indicate patches to be masked (ignored)
    - False values indicate patches to be processed or predicted

    This differs from IJEPAMaskCollator which uses integer indices for masked patches.
    """

    def __init__(
        self,
        input_size: size_2d,
        patch_size: size_2d,
        mask_scale: tuple[float, float] = (0.10, 0.25),  # masking 1/9 ~ 1/4 region
        n_masks: int = 4,
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        min_keep: int = 10,
    ) -> None:
        """Initialize the BoolIJEPAMultiBlockMaskCollator.

        Args:
            input_size (size_2d): Size of the input image.
            patch_size (size_2d): Size of each patch.
            mask_scale (tuple[float, float]): Range of mask scale (min, max).
            n_masks (int): Number of mask candidates to generate.
            aspect_ratio (tuple[float, float]): Range of aspect ratios for masks.
            min_keep (int): Minimum number of patches to keep unmasked.
        """
        super().__init__()
        assert mask_scale[0] < mask_scale[1]
        assert mask_scale[0] > 0
        assert mask_scale[1] < 1

        input_size = size_2d_to_int_tuple(input_size)
        self.patch_size = size_2d_to_int_tuple(patch_size)

        assert input_size[0] % self.patch_size[0] == 0
        assert input_size[1] % self.patch_size[1] == 0

        self.n_patches_height = input_size[0] // self.patch_size[0]
        self.n_patches_width = input_size[1] // self.patch_size[1]
        assert min_keep <= self.n_patches_height * self.n_patches_width

        self.mask_scale = mask_scale
        self.n_masks = n_masks
        self.aspect_ratio = aspect_ratio
        self.min_keep = min_keep  # minimum number of patches to keep unmasked
        self._itr_counter = Value("i", random.randrange(2**32))  # collator is shared across worker processes

    @property
    def n_patches(self) -> int:
        return self.n_patches_height * self.n_patches_width

    def step(self) -> int:
        """Increment and return the iteration counter."""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_mask_rectangle(
        self,
        generator: torch.Generator,
    ) -> tuple[int, int, int, int]:
        """Randomly sample a rectangular mask.

        Args:
            generator (torch.Generator): Generator for pseudo-random numbers.

        Returns:
            tuple[int, int, int, int]: Top, bottom, left, and right coordinates of the mask.
        """
        _scale_rand, _ratio_rand = torch.rand(2, generator=generator).tolist()
        # -- Sample mask scale
        min_s, max_s = self.mask_scale
        mask_scale = min_s + _scale_rand * (max_s - min_s)
        max_keep = mask_scale * self.n_patches

        # -- Sample mask aspect-ratio
        min_ar, max_ar = self.aspect_ratio
        aspect_ratio = min_ar + _ratio_rand * (max_ar - min_ar)

        # -- Compute height and width of mask (given scale and aspect-ratio)
        patch_ar = self.n_patches_width / self.n_patches_height
        if patch_ar > aspect_ratio:
            h_max = self.n_patches_height
            w_max = self.n_patches_height * aspect_ratio
        else:
            h_max = self.n_patches_width / aspect_ratio
            w_max = self.n_patches_width

        num_patches_max = h_max * w_max
        scale = math.sqrt(max_keep / num_patches_max)
        h, w = round(scale * h_max), round(scale * w_max)

        # Apply min keep
        if h * w < self.min_keep:
            scale = math.sqrt(self.min_keep / num_patches_max)
            h, w = math.ceil(scale * h_max), math.ceil(scale * w_max)

        # clamp
        h, w = min(max(h, 1), self.n_patches_height), min(max(w, 1), self.n_patches_width)

        # -- Compute mask coordinates
        top = int(torch.randint(high=self.n_patches_height - h + 1, size=(1,), generator=generator).item())
        left = int(torch.randint(high=self.n_patches_width - w + 1, size=(1,), generator=generator).item())
        bottom = top + h
        right = left + w
        return top, bottom, left, right

    def sample_masks_and_target(self, generator: torch.Generator) -> tuple[Tensor, Tensor]:
        """Sample boolean masks for the encoder and a target mask for the
        predictor.

        Args:
            generator (torch.Generator): Generator for pseudo-random numbers.

        Returns:
            tuple[Tensor, Tensor]:
                - encoder_mask: Boolean mask for the encoder (True for masked patches)
                - predictor_target: Boolean mask representing the target for the predictor (True for masked patches)
        """
        sampled_masks = []
        for _ in range(self.n_masks):
            mask = torch.zeros(self.n_patches_height, self.n_patches_width, dtype=torch.bool)
            top, bottom, left, right = self._sample_mask_rectangle(generator)
            mask[top:bottom, left:right] = True
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

    def __call__(self, images: list[tuple[Tensor]]) -> tuple[Tensor, Tensor, Tensor]:
        """Collate input images and create boolean masks for context encoder
        and predictor target.

        Args:
            images (list[tuple[Tensor]]): List of image tensors. Each image is shape [3, height, width].

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                - collated_images: Collated images (shape: [batch_size, 3, height, width])
                - collated_encoder_masks: Boolean masks for context encoder (shape: [batch_size, n_patches])
                - collated_predictor_targets: Boolean masks representing predictor targets (shape: [batch_size, n_patches])
        """
        collated_images: Tensor = default_collate(images)[0]

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        encoder_masks, predictor_targets = [], []
        for _ in range(len(images)):
            enc_mask, pred_target = self.sample_masks_and_target(g)
            encoder_masks.append(enc_mask)
            predictor_targets.append(pred_target)

        collated_encoder_masks = torch.stack(encoder_masks)
        collated_predictor_targets = torch.stack(predictor_targets)

        return (
            collated_images,
            collated_encoder_masks,
            collated_predictor_targets,
        )
