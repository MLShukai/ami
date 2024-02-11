import logging

import pytest

from ami.threads.base_thread import BaseThread
from ami.threads.shared_object_pool import SharedObjectPool
from ami.threads.thread_types import ThreadTypes


class MainThreadImpl(BaseThread):
    THREAD_TYPE = ThreadTypes.MAIN

    def worker(self) -> None:
        self.logger.info("worker")

    def on_shared_object_pool_attached(self) -> None:
        self.share_object("a", "object")


class ThreadImplWithError(BaseThread):
    THREAD_TYPE = ThreadTypes.MAIN

    def worker(self) -> None:
        raise RuntimeError("runtime error from worker.")


class TestBaseThread:
    def test_not_implemented_thread_type(self) -> None:
        with pytest.raises(NotImplementedError):

            class _(BaseThread):
                pass

    def test_run(self, caplog: pytest.LogCaptureFixture):
        t = ThreadImplWithError()
        with pytest.raises(RuntimeError):
            t.run()

        e = caplog.records[0]
        assert e.message == "An exception occurred in the worker thread."
        assert e.levelno == logging.ERROR

    def test_object_sharing(self):
        t = MainThreadImpl()
        sop = SharedObjectPool()
        t.attach_shared_object_pool(sop)

        assert t.get_shared_object(ThreadTypes.MAIN, "a") == "object"
