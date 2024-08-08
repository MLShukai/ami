# Ref: https://github.com/facebookresearch/ijepa

import math
import random
from multiprocessing import Value
from typing import Optional

import torch

from ami.logger import get_training_thread_logger

logger = get_training_thread_logger(__file__)


class IJEPAMultiBlockMaskCollator:
    def __init__(
        self,
        input_size: tuple[int, int] | int = (224, 224),
        patch_size: int = 16,
        encoder_mask_scale: tuple[float, float] = (0.85, 1.0),
        predictor_mask_scale: tuple[float, float] = (0.15, 0.2),
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        n_masks_for_context_encoder: int = 1,
        n_masks_for_predictor: int = 4,
        min_keep: int = 10,
        allow_overlap: bool = False,
    ) -> None:
        """Collator for dataloader during training I-JEPA.

        Collate input images and create masks for context_encoder and
        predictor.
        """
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        assert input_size[0] % patch_size == 0 and input_size[1] % patch_size == 0
        self.height, self.width = (
            input_size[0] // patch_size,
            input_size[1] // patch_size,
        )
        self.encoder_mask_scale = encoder_mask_scale
        self.predictor_mask_scale = predictor_mask_scale
        self.aspect_ratio = aspect_ratio
        self.n_masks_for_context_encoder = n_masks_for_context_encoder
        self.n_masks_for_predictor = n_masks_for_predictor
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap between encoder and predictor masks
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes

    def step(self) -> int:
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_mask_size(
        self,
        generator: torch.Generator,
        scale_range: tuple[float, float],
        aspect_ratio_range: tuple[float, float],
    ) -> tuple[int, int]:
        """randomly sampling mask's size.

        Args:
            generator (torch.Generator): Generator to make pseudo random numbers.
            scale_range (tuple[float, float]): Min and max values of mask scale.
            aspect_ratio_range (tuple[float, float]): Min and max values of aspect ratio.

        Returns:
            tuple[int, int]: Height and width of mask about to create.
        """
        _scale_rand, _ratio_rand = torch.rand(2, generator=generator).tolist()
        # -- Sample mask scale
        min_s, max_s = scale_range
        mask_scale = min_s + _scale_rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample mask aspect-ratio
        min_ar, max_ar = aspect_ratio_range
        aspect_ratio = min_ar + _ratio_rand * (max_ar - min_ar)
        # -- Compute height and width of mask (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        h = min(self.height, h)
        w = min(self.width, w)

        return (h, w)

    def _sample_mask(
        self,
        mask_size: tuple[int, int],
        acceptable_regions: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """make masks.

        Args:
            mask_size (tuple[int, int]):
                mask size (height and width).
            acceptable_regions (list[torch.Tensor] | None):
                Represents areas that may be masked.
                Necessary to prevent masks for context encoder from being covered by the masks for predictor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                mask (shape: [range(self.min_keep to self.height * self.width)]) and
                mask_complement (shape: [self.height, self.width]).
        """
        h, w = mask_size

        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample mask's top-left corner
            top = random.randint(0, self.height - h)  # including (self.height - h)
            left = random.randint(0, self.width - w)  # including (self.width - w)
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                N = max(int(len(acceptable_regions) - tries), 0)
                for k in range(N):
                    mask *= acceptable_regions[k]

            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(
                        f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"'
                    )
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0
        # --
        return mask, mask_complement

    def __call__(
        self, images: list[tuple[torch.Tensor]]
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Collate input images and create masks for context_encoder and
        predictor.

        Args:
            images (list[tuple[torch.Tensor]]):
                images list. len(images)==batch_size.
                Each image is shape [3, height, width]

        Returns:
            tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
                collated_images (torch.Tensor):
                    Collated images (shape is [batch_size, 3, height, width])
                collated_masks_for_context_encoder (list[torch.Tensor]):
                    Collated mask indices patch for context encoder.
                    (Each Tensor's shape is [batch_size, n_patch_to_keep].)
                collated_masks_for_predictor (list[torch.Tensor]):
                    Collated mask indices patch for predictor.
                    (Each Tensor's shape is [batch_size, n_patch_to_keep].)
        """
        B = len(images)

        collated_images: torch.Tensor = torch.utils.data.default_collate(images)[0]

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        # randomly sampling size of mask for predictor
        mask_size_for_predictor: tuple[int, int] = self._sample_mask_size(
            generator=g,
            scale_range=self.predictor_mask_scale,
            aspect_ratio_range=self.aspect_ratio,
        )
        # randomly sampling size of mask for context_encoder
        mask_size_for_context_encoder: tuple[int, int] = self._sample_mask_size(
            generator=g,
            scale_range=self.encoder_mask_scale,
            aspect_ratio_range=(1.0, 1.0),
        )

        # make masks
        masks_list_for_predictor: list[list[torch.Tensor]] = []
        masks_list_for_context_encoder: list[list[torch.Tensor]] = []
        min_keep_predictor = self.height * self.width
        min_keep_encoder = self.height * self.width
        for _ in range(B):
            # create mask for predictor and mask to constrain range
            masks_for_predictor: list[torch.Tensor] = []
            masks_complement: list[torch.Tensor] = []
            for _ in range(self.n_masks_for_predictor):
                mask, mask_complement = self._sample_mask(mask_size=mask_size_for_predictor)
                masks_for_predictor.append(mask)
                masks_complement.append(mask_complement)
                min_keep_predictor = min(min_keep_predictor, len(mask))
            masks_list_for_predictor.append(masks_for_predictor)

            acceptable_regions: Optional[list[torch.Tensor]] = masks_complement
            if self.allow_overlap:
                acceptable_regions = None

            # create mask for context_encoder
            masks_for_context_encoder: list[torch.Tensor] = []
            for _ in range(self.n_masks_for_context_encoder):
                mask, _ = self._sample_mask(
                    mask_size=mask_size_for_context_encoder,
                    acceptable_regions=acceptable_regions,
                )
                masks_for_context_encoder.append(mask)
                min_keep_encoder = min(min_keep_encoder, len(mask))
            masks_list_for_context_encoder.append(masks_for_context_encoder)

        # collate masks for predictor
        collated_masks_for_predictor: list[torch.Tensor] = torch.utils.data.default_collate(
            [[cm[:min_keep_predictor] for cm in cm_list] for cm_list in masks_list_for_predictor]
        )
        # collate masks for context_encoder
        collated_masks_for_context_encoder: list[torch.Tensor] = torch.utils.data.default_collate(
            [[cm[:min_keep_encoder] for cm in cm_list] for cm_list in masks_list_for_context_encoder]
        )

        return (
            collated_images,
            collated_masks_for_context_encoder,
            collated_masks_for_predictor,
        )
