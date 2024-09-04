import threading
from typing import Generator

import bottle
import pytest
import requests
from pytest import LogCaptureFixture
from webtest import TestApp

from ami.threads.inference_thread import InferenceThread
from ami.threads.main_thread import MainThread
from ami.threads.training_thread import TrainingThread


@pytest.fixture
def webapp() -> Generator[TestApp, None, None]:
    webapp = bottle.default_app()

    yield TestApp(webapp)

    webapp.reset()


def test_run_threads(thread_objects: tuple[MainThread, InferenceThread, TrainingThread], webapp: TestApp) -> None:
    mt, it, tt = thread_objects

    def send_shutdown():
        requests.post("http://127.0.0.1:44154/api/shutdown")

    it.start()
    tt.start()

    threading.Timer(1.0, send_shutdown).start()
    mt.run()

    it.join()
    tt.join()


def test_shutdown_by_max_uptime(
    thread_objects: tuple[MainThread, InferenceThread, TrainingThread],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    mt, it, tt = thread_objects

    it.start()
    tt.start()
    mt._max_uptime = 0.5
    mt.run()

    it.join()
    tt.join()

    assert "Shutting down by reaching maximum uptime." in caplog.messages


def test_setting_exception_flag(caplog: LogCaptureFixture, thread_objects, mocker) -> None:
    caplog.set_level("INFO")
    mt, it, tt = thread_objects
    mt: MainThread
    it: InferenceThread
    tt: TrainingThread

    def exception_worker():
        raise Exception

    it.worker = exception_worker
    tt.worker = exception_worker
    mt.save_checkpoint = mocker.Mock()  # For avoid pausing after exception.

    it.start()
    tt.start()

    mt.run()

    assert it.exception_flag.is_raised()
    assert tt.exception_flag.is_raised()

    assert "An exception occurred. The system will terminate immediately." in caplog.messages
