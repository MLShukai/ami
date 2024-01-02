"""This file contains a base thread class, its subclasses (for other base
thread classes), and some tools.."""
import threading
from abc import ABC
from collections import OrderedDict
from enum import Enum, auto
from typing import Any


class ThreadTypes(Enum):
    """Enumerates all types of threads that are implemented."""

    MAIN = auto()
    INFERENCE = auto()
    TRAINING = auto()


class SharedObjectPool:
    """Contains shared objects from all types of threads.

    All objects shared between threads are contained in this class.
    """

    _objects: OrderedDict[ThreadTypes, OrderedDict[str, Any]]

    def __init__(self) -> None:
        """Constructs this class."""
        self._objects = OrderedDict()
        for e in ThreadTypes:
            self._objects[e] = OrderedDict()

        # Ensure thread safety in this class at the register or get methods.
        self._lock = threading.RLock()

    def register(self, thread_type: ThreadTypes, name: str, obj: Any) -> None:
        """Register shared object.

        Args:
            thread_type: Thread type that shares object.
            name: Object name.
            obj: Actual object.
        """
        with self._lock:
            self._objects[thread_type][name] = obj

    def get(self, thread_type: ThreadTypes, name: str) -> Any:
        """Get shared object.

        Args:
            thread_type: The thread type that shared the object to be retrieved.
            name: Object name.
        """
        with self._lock:
            return self._objects[thread_type][name]


class BaseThread(ABC):
    """Base class for all thread objects.

    You must define the `_thread_type` attribute in the subclass's class field.
    Please override the :meth:`worker` method for the thread's program.

    To share objects between threads, override the :meth:`on_shared_object_pool_attached` method and use the :meth:`share_object` method.
    To get a shared object, use the :meth:`get_shared_object` method.

    NOTE: Cannot create multiple threads of the same type due to competition in the value sharing namespace.
    """

    _thread_type: ThreadTypes
    _shared_object_pool: SharedObjectPool

    def __init__(self) -> None:
        self._worker_thread = threading.Thread(target=self.worker)

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "_thread_type"):
            raise NotImplementedError("Thread class must define `_thread_type` attribute.")

    @property
    def thread_type(self) -> ThreadTypes:
        """thread type is readonly variable."""
        return self._thread_type

    def worker(self) -> None:
        """The program for this thread.

        please override this method.
        """
        raise NotImplementedError

    def start(self) -> None:
        self._worker_thread.start()

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
        self._shared_object_pool.register(self.thread_type, name, obj)

    def get_shared_object(self, shared_from: ThreadTypes, name: str) -> Any:
        """Gets the shared object.

        Args:
            shared_from: The thread type that shared the object to be retrieved.
            name: Object name.
        """
        return self._shared_object_pool.get(shared_from, name)


class BaseMainThread(BaseThread):
    """Base class of main thread."""

    _thread_type = ThreadTypes.MAIN


class BaseInferenceThread(BaseThread):
    """Base class of inference thread."""

    _thread_type = ThreadTypes.INFERENCE


class BaseTrainingThread(BaseThread):
    """Base class of training thread."""

    _thread_type = ThreadTypes.TRAINING
