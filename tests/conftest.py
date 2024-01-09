"""This file contains the pytest fixtures."""
import pytest
import torch

from tests.helpers import get_gpu_device


@pytest.fixture
def gpu_device() -> torch.device | None:
    """Fixture for retrieving the available gpu device."""
    return get_gpu_device()
