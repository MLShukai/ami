import logging

import pytest

from ami.threads.base_thread import BaseThread
from ami.threads.thread_types import ThreadTypes


class MainThreadImpl(BaseThread):
    THREAD_TYPE = ThreadTypes.MAIN

    def worker(self) -> None:
        self.logger.info("worker")


class ThreadImplWithError(BaseThread):
    THREAD_TYPE = ThreadTypes.MAIN

    def worker(self) -> None:
        raise RuntimeError("runtime error from worker.")


class TestBaseThread:
    def test_not_implemented_thread_type(self) -> None:
        with pytest.raises(NotImplementedError):

            class T(BaseThread):
                pass

            T()

    def test_run(self, caplog: pytest.LogCaptureFixture):
        t = ThreadImplWithError()
        with pytest.raises(RuntimeError):
            t.run()

        e = caplog.records[0]
        assert e.message == "An exception occurred in the worker thread."
        assert e.levelno == logging.ERROR
