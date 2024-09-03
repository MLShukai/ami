from pathlib import Path
from typing import Any

import torch
import torchvision.io as io
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset


class IntervalSamplingImageDataset(Dataset[tuple[torch.Tensor]]):
    """Dataset class for sampling images at regular intervals from a directory.

    This class creates a dataset by sampling images at fixed intervals from a specified directory.
    It supports various image formats and can optionally pre-load images into memory for faster access.

    Key Features:
    - Samples images at regular intervals from a directory.
    - Supports multiple image formats (jpeg, jpg, png by default).
    - Applies specified transformations to the images.
    - Option to pre-load images into memory for faster access.

    Usage:
    - Initialize the dataset with the desired parameters.
    - Use it with PyTorch DataLoader for efficient batch processing.

    Example:
        transform = v2.Compose([v2.Resize((256, 256)), v2.ToTensor()])
        dataset = IntervalSamplingImageDataset("/path/to/images", transform, num_sample=1000)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    Note:
    - Ensure the image directory contains only image files of the specified extensions.
    - Pre-loading images can significantly increase memory usage for large datasets.
    """

    def __init__(
        self,
        image_dir: str | Path,
        transform: v2.Transform,
        num_sample: int,
        extensions: tuple[str, ...] = ("jpeg", "JPEG", "jpg", "JPG", "png", "PNG"),
        pre_loading: bool = True,
    ) -> None:
        """Initializes the IntervalSamplingImageDataset.

        Args:
            image_dir: Directory containing the images.
            transform: Transformations to apply to the images.
            num_sample: Target number of samples to extract.
            extensions: Tuple of allowed image file extensions.
            pre_loading: If True, pre-loads all images into memory.
        """
        super().__init__()
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.extensions = extensions
        available_image_files = self._list_image_files()
        self._interval = max(len(available_image_files) // num_sample, 1)
        self.image_files = self._sample_image_files(available_image_files, self._interval)

        self.image_data: list[torch.Tensor] | None = None
        if pre_loading:
            self.image_data = [self._read_image(f) for f in self.image_files]

    def _list_image_files(self) -> list[Path]:

        files: list[Path] = []
        for ext in self.extensions:
            files.extend(self.image_dir.glob(f"*.{ext}"))

        return sorted(files)

    @staticmethod
    def _sample_image_files(image_files: list[Path], interval: int) -> list[Path]:
        files = []
        for i, file in enumerate(image_files):
            if i % interval == 0:
                files.append(file)
        return files

    def _read_image(self, file: Path) -> torch.Tensor:
        image = io.image.read_image(str(file))
        return self.transform(image)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        if self.image_data is None:
            image = self._read_image(self.image_files[index])
        else:
            image = self.image_data[index]
        return (image,)  # for adjusting batch format to `TensorDataset`.


class NormalizeToMean0Std1(v2.Transform):
    """Normalize input tensor to mean 0 and std 1."""

    def _transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        inpt = inpt.float()
        mean = inpt.mean()
        std = inpt.std()
        return (inpt - mean) / (std + 1e-6)
