import threading
from collections import OrderedDict
from enum import StrEnum, auto
from typing import Any

from .thread_types import ThreadTypes


class SharedObjectNames(StrEnum):
    """Enumerates shared object names for the purpose of registering and
    retrieving shared objects within a system."""

    THREAD_COMMAND_HANDLERS = auto()


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
