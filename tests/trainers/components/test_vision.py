from pathlib import Path

import pytest
import torch
import torchvision.transforms.v2 as v2
from torchvision.io import write_jpeg

from ami.trainers.components.vision import IntervalSamplingImageDataset


class TestIntervalSamplingImageDataset:
    @pytest.fixture
    def sample_image_dir(self, tmp_path):
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        for i in range(10):
            img = torch.zeros(3, 64, 64, dtype=torch.uint8)
            write_jpeg(img, str(image_dir / f"img_{i}.jpg"))
        return image_dir

    @pytest.fixture
    def dataset(self, sample_image_dir):
        transform = v2.Compose([v2.Resize((32, 32)), v2.ToTensor()])
        return IntervalSamplingImageDataset(sample_image_dir, transform, num_sample=5)

    def test_init(self, sample_image_dir):
        transform = v2.Compose([v2.Resize((32, 32)), v2.ToTensor()])
        dataset = IntervalSamplingImageDataset(sample_image_dir, transform, num_sample=5)
        assert len(dataset.image_files) == 5
        assert dataset._interval == 2

    def test_len(self, dataset):
        assert len(dataset) == 5

    def test_getitem(self, dataset):
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 1
        assert item[0].shape == (3, 32, 32)
        assert torch.allclose(item[0], torch.zeros(3, 32, 32, dtype=torch.uint8))

    def test_pre_loading(self, sample_image_dir):
        transform = v2.Compose([v2.Resize((32, 32)), v2.ToTensor()])
        dataset = IntervalSamplingImageDataset(sample_image_dir, transform, num_sample=5, pre_loading=True)
        assert dataset.image_data is not None
        assert len(dataset.image_data) == 5
        assert all(img.shape == (3, 32, 32) for img in dataset.image_data)

    def test_without_pre_loading(self, sample_image_dir):
        transform = v2.Compose([v2.Resize((32, 32)), v2.ToTensor()])
        dataset = IntervalSamplingImageDataset(sample_image_dir, transform, num_sample=5, pre_loading=False)
        assert dataset.image_data is None
        assert all(dataset[i][0].shape == (3, 32, 32) for i in range(len(dataset)))
