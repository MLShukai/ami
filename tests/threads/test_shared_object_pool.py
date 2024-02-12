from ami.threads.shared_object_pool import SharedObjectPool
from ami.threads.thread_types import ThreadTypes


class TestSharedObjectPool:
    def test_get_and_register(self) -> None:
        pool = SharedObjectPool()

        pool.register(ThreadTypes.MAIN, "int", 0)
        pool.register(ThreadTypes.INFERENCE, "string", "hello, world!")
        pool.register(ThreadTypes.TRAINING, "empty_list", [])

        assert pool.get(ThreadTypes.MAIN, "int") == 0
        assert pool.get(ThreadTypes.INFERENCE, "string") == "hello, world!"
        assert pool.get(ThreadTypes.TRAINING, "empty_list") == []
