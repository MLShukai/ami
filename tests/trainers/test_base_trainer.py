import pytest

from ami.data.utils import DataUsersDict
from ami.models.utils import ModelsDict
from ami.trainers.base_trainer import BaseTrainer


class TrainerImpl(BaseTrainer):
    def train(self) -> None:
        model1 = self.get_training_model("model1")
        model2 = self.get_frozen_model("model2")

        data_user = self.get_data_user("buffer1")


class TestTrainer:
    @pytest.fixture
    def trainer(self, models_dict: ModelsDict, data_users_dict: DataUsersDict) -> TrainerImpl:
        trainer = TrainerImpl()
        trainer.attach_models_dict(models_dict)
        trainer.attach_inferences_dict(models_dict.create_inferences())
        trainer.attach_data_users_dict(data_users_dict)
        return trainer

    def test_run(self, trainer: TrainerImpl) -> None:
        trainer.run()

    def test_get_frozen_model(self, trainer: TrainerImpl) -> None:
        m = trainer.get_frozen_model("model1")
        for p in m.parameters():
            assert not p.requires_grad

    def test_get_training_model(self, trainer: TrainerImpl) -> None:
        m = trainer.get_training_model("model1")
        for p in m.parameters():
            assert p.requires_grad
        assert "model1" in trainer._synchronized_model_names

    def test_sync_a_model(self, trainer: TrainerImpl) -> None:
        old_model, old_inference_model = trainer._models_dict["model1"], trainer._inferences_dict["model1"].model
        trainer._sync_a_model("model1")
        new_model, new_inference_model = trainer._models_dict["model1"], trainer._inferences_dict["model1"].model

        assert old_model is new_inference_model
        assert new_model is old_inference_model
