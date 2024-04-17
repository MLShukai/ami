from pathlib import Path

from pytest_mock import MockerFixture

from ami.threads.inference_thread import InferenceThread, Interaction


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
