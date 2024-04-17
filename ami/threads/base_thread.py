from abc import ABC
from collections import OrderedDict
from pathlib import Path
from typing import Any, TypeAlias

from ami.checkpointing import SaveAndLoadStateMixin

from ..logger import get_thread_logger
from .thread_types import ThreadTypes, get_thread_name_from_type

SharedObjectsDictType: TypeAlias = OrderedDict[str, Any]
SharedObjectsPoolType: TypeAlias = OrderedDict[ThreadTypes, SharedObjectsDictType]


class BaseThread(ABC, SaveAndLoadStateMixin):
    """Base class for all thread objects.

    You must define the `THREAD_TYPE` attribute in the subclass's class field.
    Override the :meth:`worker` method for the thread's program.

    Use `share_object` in the constructor to share objects between threads.
    Override `on_shared_object_pool_attached` and use `get_shared_object` to retrieve shared objects.

    NOTE: Cannot create multiple threads of the same type due to competition in the value sharing namespace.
    """

    THREAD_TYPE: ThreadTypes
    _shared_objects_pool: SharedObjectsPoolType

    @property
    def thread_name(self) -> str:
        """Retrieves thread name from `THREAD_TYPE`."""
        return get_thread_name_from_type(self.THREAD_TYPE)

    def __init__(self) -> None:
        """Initializes the thread and sets up logging.

        Initializes shared objects for inter-thread communication.
        """
        if not hasattr(self, "THREAD_TYPE"):
            raise NotImplementedError("Subclasses must define a `THREAD_TYPE` attribute.")

        self.logger = get_thread_logger(self.THREAD_TYPE, self.__class__.__name__)
        self.shared_objects_from_this_thread: SharedObjectsDictType = OrderedDict()

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

    def attach_shared_object_pool(self, shared_object_pool: SharedObjectsPoolType) -> None:
        """Attaches a shared object pool for inter-thread communication."""
        self._shared_objects_pool = shared_object_pool
        self.on_shared_objects_pool_attached()

    def on_shared_objects_pool_attached(self) -> None:
        """Callback for when the shared object pool is attached.

        Override for custom behavior. Use `get_shared_object` to
        retrieve shared objects.
        """

    def share_object(self, name: str, obj: Any) -> None:
        """Shares an object with other threads.

        Args:
            name: The object's name.
            obj: The object to share.
        """
        if not hasattr(self, "shared_objects_from_this_thread"):
            raise AttributeError("Call `BaseThread.__init__()` before sharing objects.")
        self.shared_objects_from_this_thread[name] = obj

    def get_shared_object(self, shared_from: ThreadTypes, name: str) -> Any:
        """Retrieves a shared object.

        Args:
            shared_from: The thread type sharing the object.
            name: The object's name.
        """
        return self._shared_objects_pool[shared_from][name]


def attach_shared_objects_pool_to_threads(*threads: BaseThread) -> None:
    """Combines and attaches shared objects from all threads to each thread.

    Args:
        *threads: The threads to process.

    Raises:
        RuntimeError: If duplicate thread types are detected.
    """
    shared_objects_pool: SharedObjectsPoolType = OrderedDict()

    for thread in threads:
        if thread.THREAD_TYPE in shared_objects_pool:
            raise RuntimeError(f"Duplicate thread type: '{thread.THREAD_TYPE}'. Each thread type must be unique.")
        shared_objects_pool[thread.THREAD_TYPE] = thread.shared_objects_from_this_thread

    for thread in threads:
        thread.attach_shared_object_pool(shared_objects_pool)
