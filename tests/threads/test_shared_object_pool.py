from ami.threads.shared_object_pool import SharedObjectPool
from ami.threads.thread_types import ThreadTypes


def test_register_and_get():
    sop = SharedObjectPool()
    sop.register(ThreadTypes.MAIN, "a", [0])
    assert sop.get(ThreadTypes.MAIN, "a") == [0]
