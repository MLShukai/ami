# Ref: https://github.com/facebookresearch/ijepa

from typing import List, Optional, Tuple

import math
import torch
from multiprocessing import Value
from logging import getLogger

logger = getLogger()

class IJEPAMaskCollator:
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        n_masks_for_context_encoder=1,
        n_masks_for_predictor=4,
        min_keep=10,
        allow_overlap=False,
    ) -> None:
        """
        """
        super(IJEPAMaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        assert input_size[0]%patch_size==0 and input_size[1]%patch_size==0
        self.height, self.width = (
            input_size[0] // patch_size,
            input_size[1] // patch_size,
        )
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.n_masks_for_context_encoder = n_masks_for_context_encoder
        self.n_masks_for_predictor = n_masks_for_predictor
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = (
            allow_overlap  # whether to allow overlap b/w enc and pred masks
        )
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_mask_size(
        self,
        generator: torch.Generator,
        scale_range: Tuple[float, float],
        aspect_ratio_range: Tuple[float, float],
    ) -> Tuple[int, int]:
        """randomly sampling mask's size.

        Args:
            generator (torch.Generator): Generator to make pseudo random numbers.
            scale_range (Tuple[float, float]): Min and max values of mask scale.
            aspect_ratio_range (Tuple[float, float]): Min and max values of aspect ratio.

        Returns:
            Tuple[int, int]: Height and width of mask about to create.
        """
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample mask scale
        min_s, max_s = scale_range
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample mask aspect-ratio
        min_ar, max_ar = aspect_ratio_range
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute height and width of mask (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_mask(
        self,
        mask_size: Tuple[int, int],
        acceptable_regions: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """make masks.

        Args:
            mask_size (Tuple[int, int]): mask size (height and width).
            acceptable_regions (List[torch.Tensor] or None):

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                mask (shape: [self.height * self.width]) and
                mask_complement (shape: [self.height, self.width]).
        """
        h, w = mask_size

        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample mask's top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
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
        self, images: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Collate input images and create masks for context_encoder and predictor.

        Args:
            images (List[torch.Tensor]): 
                images list. len(images)==batch_size. 
                Each image is shape [3, height, width]

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
                collated_images (torch.Tensor): 
                    Collated images (shape is [batch_size, 3, height, width])
                collated_masks_for_context_encoder (List[torch.Tensor]):
                    Collated mask indices patch for context encoder.
                    (Each Tensor's shape is [batch_size, n_patch_to_keep].)
                collated_masks_for_predictor (List[torch.Tensor]):
                    Collated mask indices patch for predictor.
                    (Each Tensor's shape is [batch_size, n_patch_to_keep].)
        """
        B = len(images)

        collated_images = torch.utils.data.default_collate(images)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        # randomly sampling size of mask for predictor
        mask_size_for_predictor: Tuple[int, int] = self._sample_mask_size(
            generator=g,
            scale_range=self.pred_mask_scale,
            aspect_ratio_range=self.aspect_ratio,
        )
        # randomly sampling size of mask for context_encoder
        mask_size_for_context_encoder: Tuple[int, int] = self._sample_mask_size(
            generator=g,
            scale_range=self.enc_mask_scale,
            aspect_ratio_range=(1.0, 1.0),
        )

        # make masks
        collated_masks_for_predictor: List[List[torch.Tensor]] = []
        collated_masks_for_context_encoder: List[List[torch.Tensor]] = []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):
            # create mask for predictor and mask to constrain range
            masks_for_predictor: List[torch.Tensor] = []
            masks_complement: List[torch.Tensor] = []
            for _ in range(self.n_masks_for_predictor):
                mask, mask_complement = self._sample_mask(
                    mask_size=mask_size_for_predictor
                )
                masks_for_predictor.append(mask)
                masks_complement.append(mask_complement)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_for_predictor.append(masks_for_predictor)

            acceptable_regions: Optional[List[torch.Tensor]] = masks_complement
            try:
                if self.allow_overlap:
                    acceptable_regions = None
            except Exception as e:
                logger.warning(f"Encountered exception in mask-generator {e}")

            # create mask for context_encoder
            masks_for_context_encoder: List[torch.Tensor] = []
            for _ in range(self.n_masks_for_context_encoder):
                mask, _ = self._sample_mask(
                    mask_size=mask_size_for_context_encoder,
                    acceptable_regions=acceptable_regions,
                )
                masks_for_context_encoder.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_for_context_encoder.append(masks_for_context_encoder)

        # collate masks for predictor
        collated_masks_for_predictor = [
            [cm[:min_keep_pred] for cm in cm_list]
            for cm_list in collated_masks_for_predictor
        ]
        collated_masks_for_predictor = torch.utils.data.default_collate(
            collated_masks_for_predictor
        )
        # collate masks for context_encoder
        collated_masks_for_context_encoder = [
            [cm[:min_keep_enc] for cm in cm_list]
            for cm_list in collated_masks_for_context_encoder
        ]
        collated_masks_for_context_encoder = torch.utils.data.default_collate(
            collated_masks_for_context_encoder
        )

        return (
            collated_images,
            collated_masks_for_context_encoder,
            collated_masks_for_predictor,
        )
