import logging

import pytest

from ami.threads.base_thread import BaseThread, attach_shared_objects_pool_to_threads
from ami.threads.thread_types import ThreadTypes


class MainThreadImpl(BaseThread):
    THREAD_TYPE = ThreadTypes.MAIN

    def __init__(self) -> None:
        super().__init__()
        self.share_object("a", "object")

    def worker(self) -> None:
        self.logger.info("worker")


class OtherThreadImpl(BaseThread):
    THREAD_TYPE = ThreadTypes.TRAINING  # 適当に

    def on_shared_objects_pool_attached(self) -> None:
        super().on_shared_objects_pool_attached()
        self.a = self.get_shared_object(ThreadTypes.MAIN, "a")


class ThreadImplWithError(BaseThread):
    THREAD_TYPE = ThreadTypes.MAIN

    def worker(self) -> None:
        raise RuntimeError("runtime error from worker.")


class TestBaseThread:
    def test_thread_name(self) -> None:
        assert MainThreadImpl().thread_name == "main"

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


def test_attach_shared_objects_pool_to_threads():
    mt = MainThreadImpl()
    tt = OtherThreadImpl()

    with pytest.raises(AttributeError):
        tt.a

    attach_shared_objects_pool_to_threads(mt, tt)

    assert tt.a == "object"

    # Test duplication detection.
    with pytest.raises(RuntimeError):
        attach_shared_objects_pool_to_threads(tt, tt)
