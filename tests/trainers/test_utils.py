import pytest

from ami.trainers.utils import TrainersList

from .test_base_trainer import TrainerImpl


class TestTrainersList:
    def test_get_next_trainer(self):
        trainer1 = TrainerImpl()
        trainer2 = TrainerImpl()

        trainers = TrainersList(*[trainer1, trainer2])

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

        trainers = TrainersList(*[trainer1, trainer2])
        trainers.attach_model_wrappers_dict(model_wrappers_dict)
        assert trainer1._model_wrappers_dict is model_wrappers_dict
        assert trainer2._model_wrappers_dict is model_wrappers_dict

    def test_attach_data_users_dict(self, data_users_dict):
        trainer1 = TrainerImpl()
        trainer2 = TrainerImpl()

        trainers = TrainersList(*[trainer1, trainer2])
        trainers.attach_data_users_dict(data_users_dict)
        assert trainer1._data_users_dict is data_users_dict
        assert trainer2._data_users_dict is data_users_dict

    def test_save_state(self, tmp_path):
        trainer1 = TrainerImpl()
        trainer2 = TrainerImpl()

        trainers = TrainersList(*[trainer1, trainer2])
        trainers_path = tmp_path / "trainers"
        trainers.save_state(trainers_path)
        assert trainers_path.exists()
        assert (trainers_path / "0" / "state.pt").exists()
        assert (trainers_path / "1" / "state.pt").exists()
