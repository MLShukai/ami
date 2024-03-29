import torch

from ami.models.model_wrapper import ModelWrapper
from ami.models.utils import InferenceWrappersDict, ModelWrappersDict
from tests.helpers import ModelMultiplyP, skip_if_gpu_is_not_available


class TestWrappersDict:
    @skip_if_gpu_is_not_available()
    def test_send_to_default_device(self, gpu_device: torch.device):
        mwd = ModelWrappersDict(
            a=ModelWrapper(ModelMultiplyP(), "cpu", True), b=ModelWrapper(ModelMultiplyP(), gpu_device, True)
        )

        mwd.send_to_default_device()
        assert mwd["a"].device.type == "cpu"
        assert mwd["b"].device == gpu_device

    def test_inference_wrappers_dict(self):
        mwd = ModelWrappersDict(
            a=ModelWrapper(ModelMultiplyP(), "cpu", True), b=ModelWrapper(ModelMultiplyP(), "cpu", False)
        )

        iwd = mwd.inference_wrappers_dict
        assert isinstance(iwd, InferenceWrappersDict)
        assert "a" in iwd
        assert "b" not in iwd

        iwd2 = mwd.inference_wrappers_dict
        assert iwd is iwd2
