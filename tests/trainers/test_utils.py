import pytest

from ami.trainers.utils import TrainersList

from .test_base_trainer import TrainerImpl


class TestTrainersList:
    @pytest.fixture
    def trainers(self, model_wrappers_dict, data_users_dict) -> TrainersList:
        trainers = TrainersList([TrainerImpl(), TrainerImpl()])

    def test_get_next_trainer(self):
        trainer1 = TrainerImpl()
        trainer2 = TrainerImpl()

        trainers = TrainersList([trainer1, trainer2])

        assert trainers.get_next_trainer() is trainer1
        assert trainers.get_next_trainer() is trainer2
        assert trainers.get_next_trainer() is trainer1
        assert trainers.get_next_trainer() is trainer2

        with pytest.raises(RuntimeError):
            empty_trainers = TrainersList()
            empty_trainers.get_next_trainer()

    def test_attach_model_wrappers_dict(self, model_wrappers_dict):
        trainer1 = TrainerImpl()
        trainer2 = TrainerImpl()

        trainers = TrainersList([trainer1, trainer2])
        trainers.attach_model_wrappers_dict(model_wrappers_dict)
        assert trainer1._model_wrappers_dict is model_wrappers_dict
        assert trainer2._model_wrappers_dict is model_wrappers_dict

    def test_attach_data_users_dict(self, data_users_dict):
        trainer1 = TrainerImpl()
        trainer2 = TrainerImpl()

        trainers = TrainersList([trainer1, trainer2])
        trainers.attach_data_users_dict(data_users_dict)
        assert trainer1._data_users_dict is data_users_dict
        assert trainer2._data_users_dict is data_users_dict
