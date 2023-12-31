import pytest

from ami.threads.base_threads import (
    BaseInferenceThread,
    BaseMainThread,
    BaseThread,
    BaseTrainingThread,
    SharedObjectPool,
    ThreadTypes,
)


class TestSharedObjectPool:
    def test_get_and_register(self):
        pool = SharedObjectPool()

        pool.register(ThreadTypes.MAIN, "int", 0)
        pool.register(ThreadTypes.INFERENCE, "string", "hello, world!")
        pool.register(ThreadTypes.TRAINING, "empty_list", [])

        assert pool.get(ThreadTypes.MAIN, "int") == 0
        assert pool.get(ThreadTypes.INFERENCE, "string") == "hello, world!"
        assert pool.get(ThreadTypes.TRAINING, "empty_list") == []


class TestBaseThreads:
    def test_not_implemented_thread_type(self):
        with pytest.raises(NotImplementedError):

            class _(BaseThread):
                pass

    def test_threads(self):
        main = BaseMainThread()
        infer = BaseInferenceThread()
        train = BaseTrainingThread()

        assert main.thread_type == ThreadTypes.MAIN
        assert infer.thread_type == ThreadTypes.INFERENCE
        assert train.thread_type == ThreadTypes.TRAINING

        sop = SharedObjectPool()
        main.attach_shared_object_pool(sop)
        train.attach_shared_object_pool(sop)
        infer.attach_shared_object_pool(sop)

        main.share_object("m", 0)
        assert train.get_shared_object(ThreadTypes.MAIN, "m") == 0
