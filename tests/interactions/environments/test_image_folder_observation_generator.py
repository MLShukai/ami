import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from ami.interactions.environments.image_folder_observation_generator import (
    ImageFolderObservationGenerator,
)


@pytest.fixture
def sample_image_folder(tmp_path):
    """Create a temporary folder with sample images for testing."""
    folder = tmp_path / "sample_images"
    folder.mkdir()

    # Create sample images in root and nested folders
    for i in range(2):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(folder / f"image_{i}.jpg")

    return folder


class TestImageFolderObservationGenerator:
    def test_default_transform(self, sample_image_folder):
        generator = ImageFolderObservationGenerator(sample_image_folder, (64, 64))
        result = generator()

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float
        assert result.shape == (3, 64, 64)
        assert torch.all(result >= 0) and torch.all(result <= 1)

    def test_custom_transform(self, sample_image_folder):
        custom_transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
            ]
        )
        generator = ImageFolderObservationGenerator(sample_image_folder, (64, 64), transform=custom_transform)
        result = generator()

        assert result.dtype == torch.uint8
        assert result.shape == (3, 64, 64)

    @pytest.mark.parametrize("image_size", [(32, 32), (64, 128), (224, 224)])
    def test_various_image_sizes(self, sample_image_folder, image_size):
        generator = ImageFolderObservationGenerator(sample_image_folder, image_size)
        result = generator()
        assert result.shape == (3, *image_size)

    def test_random_sampling(self, sample_image_folder):
        generator = ImageFolderObservationGenerator(sample_image_folder, (64, 64))
        samples = [generator() for _ in range(10)]

        # Check that we get different images (not guaranteed but highly likely)
        different_samples = False
        for i in range(len(samples) - 1):
            if not torch.allclose(samples[i], samples[i + 1]):
                different_samples = True
                break
        assert different_samples, "All samples are identical"
