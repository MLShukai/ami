from abc import ABC, abstractmethod
from typing import Any

from ..logger import get_thread_logger
from .shared_object_pool import SharedObjectPool
from .thread_types import ThreadTypes


class BaseThread(ABC):
    """Base class for all thread objects.

    You must define the `THREAD_TYPE` attribute in the subclass's class field.
    Override the :meth:`worker` method for the thread's program.

    To share objects between threads, override the :meth:`on_shared_object_pool_attached` method and use the :meth:`share_object` method.
    To get a shared object, use the :meth:`get_shared_object` method.

    NOTE: Cannot create multiple threads of the same type due to competition in the value sharing namespace.
    """

    THREAD_TYPE: ThreadTypes
    _shared_object_pool: SharedObjectPool

    def __init__(self) -> None:
        """Constructs the class and sets the logger."""
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
        """Executes the worker thread, logging any exceptions that occur.

        This method calls the `worker` method and logs any exceptions
        that arise during its execution. After logging, the exception is
        re-raised to ensure that exception handling can occur further up
        the call stack.
        """
        try:
            self.worker()
        except Exception:
            self.logger.exception("An exception occurred in the worker thread.")
            raise

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
