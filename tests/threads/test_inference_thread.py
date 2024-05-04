import threading
import time
from pathlib import Path

from pytest_mock import MockerFixture

from ami.interactions.interaction import Interaction
from ami.threads import InferenceThread, MainThread, TrainingThread


class TestInferenceThread:
    def test_save_and_load_state(self, thread_objects, mocker: MockerFixture, tmp_path: Path):
        inference_thread: InferenceThread = thread_objects[1]
        inference_thread.interaction = mocker.Mock(Interaction)

        inference_path = tmp_path / "inference"
        inference_thread.save_state(inference_path)
        assert inference_path.exists()
        inference_thread.interaction.save_state.assert_called_once_with(inference_path / "interaction")

        inference_thread.load_state(inference_path)
        inference_thread.interaction.load_state.assert_called_once_with(inference_path / "interaction")

    def test_pause_resume_event_callbacks(
        self, thread_objects: tuple[MainThread, InferenceThread, TrainingThread], mocker: MockerFixture
    ):
        _, inference_thread, _ = thread_objects
        inference_thread.interaction = mocker.Mock(Interaction)

        inference_thread.on_paused()
        inference_thread.interaction.on_paused.assert_called_once()

        inference_thread.on_resumed()
        inference_thread.interaction.on_resumed.assert_called_once()
