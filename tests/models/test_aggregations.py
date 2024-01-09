import torch

from ami.models.aggregations import InferencesAggregation, ModelsAggregation
from tests.helpers import skip_if_gpu_is_not_available

from .test_base_model import ModelImpl


class TestAggregations:
    @skip_if_gpu_is_not_available()
    def test_send_to_default_device(self, gpu_device: torch.device):
        ma = ModelsAggregation(a=ModelImpl("cpu", True), b=ModelImpl(gpu_device, True))

        ma.send_to_default_device()

        assert ma["a"].device.type == "cpu"
        assert ma["b"].device == gpu_device

    def test_create_inferences(self):
        ma = ModelsAggregation(a=ModelImpl("cpu", True), b=ModelImpl("cpu", False))

        ia = ma.create_inferences()
        assert isinstance(ia, InferencesAggregation)
        assert "a" in ia
        assert "b" not in ia
