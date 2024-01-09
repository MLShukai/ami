"""This file contains helper objects for testing some features."""
import pytest
import torch


def get_gpu_device() -> torch.device | None:
    """Return the available gpu device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps:0")
    else:
        return None


def skip_if_gpu_is_not_available():
    return pytest.mark.skipif(get_gpu_device() is None, reason="GPU devices are not found!")
