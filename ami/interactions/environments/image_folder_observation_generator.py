import random
from pathlib import Path

import torch
import torchvision
from torchvision.transforms import v2

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class ImageFolderObservationGenerator:
    """This class reads the images from folder and provides the tensor
    observation for `DummyEnvironment` class."""

    def __init__(
        self,
        root_dir: str | Path,
        image_size: tuple[int, int],
        transform: v2.Transform | None = None,
    ) -> None:
        """Initialize the ImageFolderObservationGenerator.

        Args:
            root_dir (str | Path): The root directory containing the image folders.
            image_size (tuple[int, int]): Target image size. height and width.
            transform (v2.Transform | None, optional): Custom transform to be applied to the images.
                If None, a default transform will be used. Defaults to None.
        """
        self.image_size = image_size
        if transform is None:
            transform = v2.Compose(
                [
                    v2.ToDtype(torch.float, scale=True),
                ]
            )
        self.transform = transform

        self._root_dir = Path(root_dir)
        self.image_files: list[Path] = []
        for ext in IMG_EXTENSIONS:
            self.image_files.extend(self._root_dir.glob("*" + ext))

    def __call__(self) -> torch.Tensor:
        """Returns a random sampled image.

        Returns:
            torch.Tensor: A tensor representing a randomly sampled image from the dataset,
                          resized to the specified image_size, with applied transformations.
        """
        i = random.randrange(len(self.image_files))
        img = torchvision.io.read_image(str(self.image_files[i]))
        img = self.transform(img)
        img = v2.functional.resize(img, self.image_size)
        return img
