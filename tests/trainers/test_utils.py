import pytest
from pytest_mock import MockerFixture

from ami.trainers.utils import BaseTrainer, TrainersList

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

    def test_save_and_load_state(self, tmp_path, mocker: MockerFixture):
        trainer1 = TrainerImpl()
        trainer2 = TrainerImpl()
        mock_trainer = mocker.Mock(BaseTrainer)

        trainers = TrainersList(*[trainer1, trainer2, mock_trainer])
        trainers_path = tmp_path / "trainers"
        trainers.save_state(trainers_path)
        assert trainers_path.exists()
        assert (trainers_path / "0" / "state.pt").exists()
        assert (trainers_path / "1" / "state.pt").exists()
        mock_trainer.save_state.assert_called_once_with(trainers_path / "2")

        trainers.load_state(trainers_path)
        mock_trainer.load_state.assert_called_once_with(trainers_path / "2")

    def test_system_event_callbacks(self, mocker: MockerFixture):
        mock_trainer = mocker.Mock(BaseTrainer)

        trainers = TrainersList(mock_trainer)

        trainers.on_paused()
        mock_trainer.on_paused.assert_called_once()

        trainers.on_resumed()
        mock_trainer.on_resumed.assert_called_once()
