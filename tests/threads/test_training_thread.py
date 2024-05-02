from pathlib import Path

from pytest_mock import MockerFixture

from ami.threads import MainThread, InferenceThread, TrainingThread
from ami.threads.training_thread import (
    DataUsersDict,
    ModelWrappersDict,
    TrainersList,
)


class TestTrainingThread:
    def test_save_and_load_state(self, thread_objects, mocker: MockerFixture, tmp_path: Path) -> TrainingThread:

        training_thread: TrainingThread = thread_objects[2]
        training_thread.data_users = mocker.Mock(DataUsersDict)
        training_thread.trainers = mocker.Mock(TrainersList)
        training_thread.models = mocker.Mock(ModelWrappersDict)

        training_path = tmp_path / "training"
        training_thread.save_state(training_path)
        assert training_path.exists()

        training_thread.data_users.save_state.assert_called_once_with(training_path / "data")
        training_thread.models.save_state.assert_called_once_with(training_path / "models")
        training_thread.trainers.save_state.assert_called_once_with(training_path / "trainers")

        training_thread.load_state(training_path)
        training_thread.data_users.load_state.assert_called_once_with(training_path / "data")
        training_thread.models.load_state.assert_called_once_with(training_path / "models")
        training_thread.trainers.load_state.assert_called_once_with(training_path / "trainers")

    def test_system_event_callbacks(
        self,
        thread_objects: tuple[MainThread, InferenceThread, TrainingThread],
        mocker: MockerFixture
    ):
        _, _, training_thread = thread_objects
        training_thread.trainers = mocker.Mock(TrainersList)

        training_thread.on_paused()
        training_thread.trainers.on_paused.assert_called_once()

        training_thread.on_resumed()
        training_thread.trainers.on_resumed.assert_called_once()
