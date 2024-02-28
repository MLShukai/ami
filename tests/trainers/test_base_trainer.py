import pytest
import torch
import torch.nn as nn

from ami.data.utils import DataUsersDict
from ami.models.model_wrapper import InferenceWrapper, ModelWrapper
from ami.models.utils import ModelWrappersDict
from tests.helpers import ModelMultiplyP, TrainerImpl


def assert_model_training(model: nn.Module) -> None:
    for p in model.parameters():
        assert p.requires_grad is True
    assert model.training is True


def assert_model_frozen(model: nn.Module) -> None:
    for p in model.parameters():
        assert p.requires_grad is False
    assert model.training is False


class TestTrainer:
    @pytest.fixture
    def trainer(self, model_wrappers_dict: ModelWrappersDict, data_users_dict: DataUsersDict) -> TrainerImpl:
        trainer = TrainerImpl()
        trainer.attach_model_wrappers_dict(model_wrappers_dict)
        trainer.attach_data_users_dict(data_users_dict)
        return trainer

    def test_run(self, trainer: TrainerImpl) -> None:
        trainer.run()

    def test_get_frozen_model_and_get_training_model(self, trainer: TrainerImpl) -> None:
        m1 = trainer.get_training_model("model1")
        m2 = trainer.get_frozen_model("model2")

        with pytest.raises(RuntimeError):
            trainer.get_frozen_model("model1")

        with pytest.raises(RuntimeError):
            trainer.get_training_model("model2")

        assert "model1" in trainer._training_model_names
        assert "model1" in trainer._synchronized_model_names
        assert "model2" in trainer._frozen_model_names
        assert "model2" not in trainer._synchronized_model_names

        trainer.get_training_model("model_no_inference")
        assert "model_no_inference" not in trainer._synchronized_model_names

        assert_model_training(m1)
        assert_model_frozen(m2)

    def test_setup(self, trainer: TrainerImpl) -> None:
        m1 = trainer.get_training_model("model1")
        m2 = trainer.get_frozen_model("model2")

        m1.freeze_model()
        m2.unfreeze_model()

        trainer.setup()

        assert_model_training(m1)
        assert_model_frozen(m2)

    def test_sync_a_model(self, trainer: TrainerImpl) -> None:
        wrapper: ModelWrapper[ModelMultiplyP] = trainer._model_wrappers_dict["model1"]
        inference: InferenceWrapper[ModelMultiplyP] = trainer._inference_wrappers_dict["model1"]

        wrapper.model.p.data += 1
        assert not torch.equal(wrapper.model.p, inference.model.p)

        trainer._sync_a_model("model1")

        assert torch.equal(wrapper.model.p, inference.model.p)

        assert_model_training(wrapper.model)
        assert_model_frozen(inference.model)
