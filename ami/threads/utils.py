import threading


class ThreadSafeFlag:
    """A thread-safe flag that can be set, cleared, or checked."""

    def __init__(self) -> None:
        """Initialize the flag as False and set up a reentrant lock."""
        self._lock = threading.RLock()
        self._flag = False

    def clear(self) -> None:
        """Clear the flag to False, thread-safely."""
        with self._lock:
            self._flag = False

    def is_set(self) -> bool:
        """Return True if the flag is set, thread-safely."""
        with self._lock:
            return self._flag

    def set(self) -> None:
        """Set the flag to True, thread-safely."""
        with self._lock:
            self._flag = True

    def take(self) -> bool:
        """Return the current value of the flag and clear it, thread-safely."""
        with self._lock:
            value = self._flag
            self.clear()
            return value
