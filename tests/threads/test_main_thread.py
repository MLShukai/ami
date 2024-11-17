import random
from pathlib import Path

import pytest

import ami.time
from ami.threads.main_thread import MainThread, ThreadTypes
from ami.threads.shared_object_names import SharedObjectNames


class TestMainThread:
    def test_object_sharing(self, checkpoint_scheduler):
        mt = MainThread(checkpoint_scheduler)

        # test can get.
        SharedObjectNames.THREAD_COMMAND_HANDLERS in mt.shared_objects_from_this_thread

    def test_pause_resume_event_callback(self, thread_objects):
        mt: MainThread = thread_objects[0]
        assert not ami.time.is_paused()
        mt.on_paused()
        assert ami.time.is_paused()
        mt.on_resumed()
        assert not ami.time.is_paused()

    def test_save_and_load_state(self, thread_objects, tmp_path: Path):
        path = tmp_path / "main"
        mt: MainThread = thread_objects[0]
        state = ami.time.state_dict()
        mt.save_state(path)
        assert path.is_dir()
        assert (path / "time_state.pkl").is_file()
        ami.time.load_state_dict(
            ami.time.TimeController.TimeControllerState(
                scaled_anchor_monotonic=random.random(),
                scaled_anchor_perf_counter=random.random(),
                scaled_anchor_time=random.random(),
            )
        )
        loaded_state = ami.time.state_dict()
        for key in loaded_state.keys():
            assert state[key] != pytest.approx(loaded_state[key], abs=0.1)

        mt.load_state(path)
        loaded_state = ami.time.state_dict()
        for key in loaded_state.keys():
            assert state[key] == pytest.approx(loaded_state[key], abs=0.1)
