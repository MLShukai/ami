import time

import pytest

from ami.threads.background_thread import BackgroundThread
from ami.threads.thread_control import ThreadCommandHandler
from ami.threads.thread_types import ThreadTypes

WAIT_TIME = 0.001


class BackgroundThreadImpl(BackgroundThread):
    THREAD_TYPE = ThreadTypes.TRAINING

    def worker(self) -> None:
        time.sleep(WAIT_TIME)


class TestBackgroundThread:
    def test_invalid_thread_type(self):
        with pytest.raises(ValueError):

            class Error(BackgroundThread):
                THREAD_TYPE = ThreadTypes.MAIN

            Error()

    def test_thread_background(self) -> None:
        bt = BackgroundThreadImpl()

        start_time = time.perf_counter()
        assert bt.is_alive() is False
        bt.start()
        assert bt.is_alive() is True
        assert time.perf_counter() - start_time < WAIT_TIME
        bt.join()
        assert time.perf_counter() - start_time > WAIT_TIME

    def test_thread_command_handler(self, thread_objects):
        _, it, tt = thread_objects
        it: BackgroundThread
        tt: BackgroundThread
        assert isinstance(it.thread_command_handler, ThreadCommandHandler)
        assert isinstance(tt.thread_command_handler, ThreadCommandHandler)
        assert it.thread_command_handler is not tt.thread_command_handler, "must be other instance."

        assert it.on_paused in it.thread_command_handler._on_paused_callbacks
        assert it.on_paused not in tt.thread_command_handler._on_paused_callbacks
        assert it.on_resumed in it.thread_command_handler._on_resumed_callbacks
        assert it.on_resumed not in tt.thread_command_handler._on_paused_callbacks

        assert tt.on_paused in tt.thread_command_handler._on_paused_callbacks
        assert tt.on_paused not in it.thread_command_handler._on_paused_callbacks
        assert tt.on_resumed in tt.thread_command_handler._on_resumed_callbacks
        assert tt.on_resumed not in it.thread_command_handler._on_paused_callbacks
