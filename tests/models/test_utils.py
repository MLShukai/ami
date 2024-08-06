import pytest
import torch

from ami.models.model_wrapper import ModelWrapper
from ami.models.utils import (
    InferenceWrappersDict,
    ModelWrappersDict,
    count_model_parameters,
    create_model_parameter_count_dict,
)
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

    def test_save_and_load_state(self, tmp_path):
        models_path = tmp_path / "models"
        mwd = ModelWrappersDict(
            a=ModelWrapper(ModelMultiplyP(), "cpu", True), b=ModelWrapper(ModelMultiplyP(), "cpu", False)
        )

        mwd.save_state(models_path)
        assert (models_path / "a.pt").exists()
        assert (models_path / "b.pt").exists()

        new_mwd = ModelWrappersDict(
            a=ModelWrapper(ModelMultiplyP(), "cpu", True), b=ModelWrapper(ModelMultiplyP(), "cpu", False)
        )

        for (wrapper, new_wrapper) in zip(mwd.values(), new_mwd.values()):
            assert not torch.equal(wrapper.model.p, new_wrapper.model.p)

        new_mwd.load_state(models_path)

        for (wrapper, new_wrapper) in zip(mwd.values(), new_mwd.values()):
            assert torch.equal(wrapper.model.p, new_wrapper.model.p)

    def test_key_alias(self, tmp_path):
        wrapper1 = ModelWrapper(ModelMultiplyP(), "cpu", True)
        wrapper2 = ModelWrapper(ModelMultiplyP(), "cpu", True)
        mwd = ModelWrappersDict(a=wrapper1, b=wrapper1)
        mwd["c"] = wrapper2
        mwd["d"] = wrapper2

        assert mwd._alias_keys == {"b", "d"}
        assert mwd._names_without_alias == {"a", "c"}

        with pytest.raises(KeyError):
            mwd["a"] = wrapper2
        with pytest.raises(RuntimeError):
            del mwd["a"]

        # Test parameter saving
        models_path = tmp_path / "models"
        mwd.save_state(models_path)
        assert (models_path / "a.pt").exists()
        assert not (models_path / "b.pt").exists()
        assert (models_path / "c.pt").exists()
        assert not (models_path / "d.pt").exists()

        mwd.load_state(models_path)


class DummyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p1 = torch.nn.Parameter(torch.randn(10))
        self.p2 = torch.nn.Parameter(torch.randn(5), requires_grad=False)


def test_count_model_parameters():
    model = DummyModel()
    assert count_model_parameters(model) == (15, 10, 5)


def test_create_model_parameter_count_dict():
    models = ModelWrappersDict(
        {
            "model1": ModelWrapper(DummyModel()),
            "model2": ModelWrapper(DummyModel()),
        }
    )

    out = create_model_parameter_count_dict(models)
    assert out["_all_"] == {
        "total": 30,
        "trainable": 20,
        "frozen": 10,
    }

    assert out["model1"] == out["model2"] == {"total": 15, "trainable": 10, "frozen": 5}
