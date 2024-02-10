from abc import ABC, abstractmethod
from typing import Any

from ..logger import get_thread_logger
from .shared_object_pool import SharedObjectPool
from .thread_types import ThreadTypes


class BaseThread(ABC):

    THREAD_TYPE: ThreadTypes
    _shared_object_pool: SharedObjectPool

    def __init__(self) -> None:
        self.logger = get_thread_logger(self.THREAD_TYPE, self.__class__.__name__)

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "THREAD_TYPE"):
            raise NotImplementedError("Thread class must define `THREAD_TYPE` attribute.")

    def worker(self) -> None:
        """The program for this thread.

        please override this method.
        """
        raise NotImplementedError

    def run(self) -> None:
        try:
            self.worker()
        except Exception as e:
            raise e

    def attach_shared_object_pool(self, shared_object_pool: SharedObjectPool) -> None:
        self._shared_object_pool = shared_object_pool
        self.on_shared_object_pool_attached()

    def on_shared_object_pool_attached(self) -> None:
        """For sharing objects, please override this callback and use
        `share_object` method."""

    def share_object(self, name: str, obj: Any) -> None:
        """Shares object to other threads.

        Args:
            name: Object name.
            obj: Actual object.
        """
        self._shared_object_pool.register(self.THREAD_TYPE, name, obj)

    def get_shared_object(self, shared_from: ThreadTypes, name: str) -> Any:
        """Gets the shared object.

        Args:
            shared_from: The thread type that shared the object to be retrieved.
            name: Object name.
        """
        return self._shared_object_pool.get(shared_from, name)
