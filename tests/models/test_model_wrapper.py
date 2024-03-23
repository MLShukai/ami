import pytest
import torch

from ami.models.model_wrapper import ModelWrapper, ThreadSafeInferenceWrapper
from tests.helpers import ModelMultiplyP, skip_if_gpu_is_not_available


class TestWrappers:
    def test_device_cpu(self):
        device = torch.device("cpu")
        m = ModelWrapper(ModelMultiplyP(), device, True)

        assert m.device == device
        m.to_default_device()
        assert m.device == m.device

    @skip_if_gpu_is_not_available()
    def test_device_gpu(self, gpu_device: torch.device):
        m = ModelWrapper(ModelMultiplyP(), gpu_device, True)

        assert m.device.type == "cpu"
        m.to_default_device()
        assert m.device == gpu_device

    def test_create_inference(self):
        m = ModelMultiplyP()
        mw = ModelWrapper(m, "cpu", True)
        inference_wrapper = mw.create_inference()
        assert isinstance(inference_wrapper, ThreadSafeInferenceWrapper)
        assert inference_wrapper.model is not m
        assert inference_wrapper.model.p is not m.p
        assert inference_wrapper.model.p == m.p

        mw = ModelWrapper(m, "cpu", False)
        with pytest.raises(RuntimeError):
            mw.create_inference()

    def test_infer_cpu(self):
        mw = ModelWrapper(ModelMultiplyP(), "cpu", True)
        mw.to_default_device()
        inference = mw.create_inference()

        data = torch.randn(10)
        assert isinstance(inference(data), torch.Tensor)

    @skip_if_gpu_is_not_available()
    def test_infer_gpu(self, gpu_device: torch.device):
        mw = ModelWrapper(ModelMultiplyP(), gpu_device, True)
        mw.to_default_device()
        inference = mw.create_inference()

        data = torch.randn(10)
        out: torch.Tensor = inference(data)
        assert out.device == gpu_device

    def test_freeze_model_and_unfreeze_model(self) -> None:
        mw = ModelWrapper(ModelMultiplyP(), "cpu", True)
        assert mw.model.p.requires_grad is True

        mw.freeze_model()
        assert mw.model.p.requires_grad is False
        assert mw.model.training is False

        mw.unfreeze_model()
        assert mw.model.p.requires_grad is True
        assert mw.model.training is True
