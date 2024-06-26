from pathlib import Path

from freezegun import freeze_time
from pytest import LogCaptureFixture
from pytest_mock import MockerFixture

from ami.checkpointing.checkpointing import Checkpointing
from ami.threads.base_thread import BaseThread


class TestCheckpointing:
    @freeze_time("2000-01-01 00:00:00")
    def test_save_and_load_checkpoint(
        self,
        tmp_path: Path,
        thread_objects,
        mocker: MockerFixture,
    ):
        ckpts_dir = tmp_path / "checkpoints"
        ckpt = Checkpointing(ckpts_dir)
        main_thread: BaseThread = thread_objects[0]
        mock_save_state = mocker.spy(main_thread, "save_state")
        mock_load_state = mocker.spy(main_thread, "load_state")
        ckpt.add_threads(main_thread)
        ckpt_path = ckpt.save_checkpoint()
        assert ckpt_path == ckpts_dir / "2000-01-01_00-00-00,000000.ckpt"
        ckpt.load_checkpoint(ckpt_path)

        main_thread_path = ckpt_path / "main"
        assert ckpts_dir.exists()
        assert ckpt_path.exists()
        mock_save_state.assert_called_once_with(main_thread_path)
        mock_load_state.assert_called_once_with(main_thread_path)

    def test_abort_save_checkpoint_to_same_dir(
        self,
        tmp_path: Path,
        caplog: LogCaptureFixture,
    ):
        name = "fixed_name.ckpt"
        ckpt = Checkpointing(tmp_path, name)
        ckpt.save_checkpoint()
        msg = f"Aborted saving checkpoint because the checkpoint path already exists.: '{tmp_path / name}.'"
        assert msg not in caplog.messages
        ckpt.save_checkpoint()
        assert msg in caplog.messages
