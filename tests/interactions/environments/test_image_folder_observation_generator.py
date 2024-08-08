import numpy as np
import pytest
import torch
from PIL import Image

from ami.interactions.environments.image_folder_observation_generator import (
    ImageFolderObservationGenerator,
)


@pytest.fixture
def sample_image_folder(tmp_path):
    folder = tmp_path / "sample_images"
    folder.mkdir()
    subfolders = ["class1", "class2"]
    for subfolder in subfolders:
        (folder / subfolder).mkdir()
        for i in range(2):
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img.save(folder / subfolder / f"image_{i}.jpg")
    return folder


class TestImageFolderObservationGenerator:
    def test_divide_255(self, sample_image_folder):
        generator = ImageFolderObservationGenerator(sample_image_folder, (64, 64), divide_255=True)
        result = generator()
        assert torch.all(result >= 0) and torch.all(result <= 1)

    def test_no_divide_255(self, sample_image_folder):
        generator = ImageFolderObservationGenerator(sample_image_folder, (64, 64), divide_255=False)
        result = generator()
        assert torch.any(result > 1)  # At least some values should be > 1 if not divided by 255

    @pytest.mark.parametrize("dtype", [torch.int, torch.double, torch.cfloat])
    def test_custom_dtype(self, sample_image_folder, dtype):
        generator = ImageFolderObservationGenerator(sample_image_folder, (64, 64), dtype=dtype)
        result = generator()
        assert result.dtype == dtype

    @pytest.mark.parametrize("image_size", [(50, 50), (100, 150), (224, 224)])
    def test_resize(self, sample_image_folder, image_size):
        generator = ImageFolderObservationGenerator(sample_image_folder, image_size)
        result = generator()
        assert result.shape == (3, *image_size)
