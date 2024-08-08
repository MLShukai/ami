import random
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import pil_to_tensor, resize


class ImageFolderObservationGenerator:
    """This class reads the images from folder and provides the tensor
    observation for `DummyEnvironment` class."""

    def __init__(
        self,
        root_dir: str | Path,
        image_size: tuple[int, int],
        divide_255: bool = True,
        dtype: torch.dtype = torch.float,
    ) -> None:
        """Initialize the ImageFolderObservationGenerator.

        Args:
            root_dir (str | Path): The root directory containing the image folders.
            image_size (tuple[int, int]): Target image size. height and width.
            divide_255 (bool, optional): Whether to divide pixel values by 255. Defaults to True.
            dtype (torch.dtype, optional): The desired data type of the output tensor. Defaults to torch.float.
        """
        self.image_size = image_size
        self.divide_255 = divide_255
        self.dtype = dtype

        self._dataset = ImageFolder(root_dir)

    def __call__(self) -> torch.Tensor:
        """Returns the random sampled image.

        Returns:
            torch.Tensor: A tensor representing a randomly sampled image from the dataset,
                          resized to the specified image_size, with optional normalization,
                          and converted to the specified dtype.
        """
        i = random.randrange(len(self._dataset))
        pil_img, _ = self._dataset[i]
        img = pil_to_tensor(pil_img)
        img = resize(img, self.image_size)
        if self.divide_255:
            img = img / 255
        img = img.type(self.dtype)
        return img
