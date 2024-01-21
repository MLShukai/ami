from typing import Any

import pytest
import torch
import torch.nn as nn

from ami.models.base_model import BaseModel, Inference
from tests.helpers import ModelImpl, skip_if_gpu_is_not_available


class TestBaseModelAndInference:
    def test_device_cpu(self):
        device = torch.device("cpu")
        m = ModelImpl(device, True)

        assert m.device == device
        m.to_default_device()
        assert m.device == device

    @skip_if_gpu_is_not_available()
    def test_device_gpu(self, gpu_device: torch.device):
        m = ModelImpl(gpu_device, True)

        assert m.device.type == "cpu"
        m.to_default_device()
        assert m.device.type == gpu_device.type
        assert m.device.index == gpu_device.index

    def test_create_inference(self):
        m = ModelImpl("cpu", True)
        inference = m.create_inference()
        assert isinstance(inference, Inference)
        assert inference.model is not m
        assert inference.model.p is not m.p
        assert inference.model.p == m.p

        m = ModelImpl("cpu", False)
        with pytest.raises(RuntimeError):
            m.create_inference()

    def test_infer_cpu(self):
        m = ModelImpl("cpu", True)
        m.to_default_device()
        inference = m.create_inference()

        data = torch.randn(10)
        assert isinstance(inference(data), torch.Tensor)

    @skip_if_gpu_is_not_available()
    def test_infer_gpu(self, gpu_device: torch.device):
        m = ModelImpl(gpu_device, True)
        m.to_default_device()
        inference = m.create_inference()

        data = torch.randn(10)
        out = inference(data)
        assert out.device.type == gpu_device.type
        assert out.device.index == gpu_device.index
