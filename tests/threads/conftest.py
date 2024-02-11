import pytest

from ami.threads.main_thread import MainThread
from ami.threads.shared_object_pool import SharedObjectNames, SharedObjectPool


@pytest.fixture
def shared_object_pool() -> SharedObjectPool:
    return SharedObjectPool()


@pytest.fixture
def main_thread(shared_object_pool) -> MainThread:
    mt = MainThread()
    mt.attach_shared_object_pool(shared_object_pool)
    return mt
