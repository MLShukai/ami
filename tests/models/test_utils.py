import torch

from ami.models.model_wrapper import ModelWrapper
from ami.models.utils import (
    InferencesDict,
    InferenceWrappersDict,
    ModelsDict,
    ModelWrappersDict,
)
from tests.helpers import ModelImpl, ModelMultiplyP, skip_if_gpu_is_not_available


class TestAggregations:
    @skip_if_gpu_is_not_available()
    def test_send_to_default_device(self, gpu_device: torch.device):
        ma = ModelsDict(a=ModelImpl("cpu", True), b=ModelImpl(gpu_device, True))

        ma.send_to_default_device()

        assert ma["a"].device.type == "cpu"
        assert ma["b"].device == gpu_device

    def test_create_inferences(self):
        ma = ModelsDict(a=ModelImpl("cpu", True), b=ModelImpl("cpu", False))

        ia = ma.create_inferences()
        assert isinstance(ia, InferencesDict)
        assert "a" in ia
        assert "b" not in ia


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
