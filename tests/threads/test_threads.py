import threading
from typing import Generator

import bottle
import pytest
import requests
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
