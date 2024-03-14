"""This file contains the interface class for a data buffer designed for multi-
threading."""
import threading
from typing import Any, Generic, TypeVar

from torch.utils.data import Dataset

from .buffers.base_data_buffer import BaseDataBuffer
from .step_data import StepData

BufferType = TypeVar("BufferType", bound=BaseDataBuffer)


class ThreadSafeDataCollector(Generic[BufferType]):
    """Collects the data in inference thread."""

    def __init__(self, buffer: BufferType) -> None:
        """Constructs data collector class."""
        self._buffer = buffer
        self._lock = threading.RLock()

    def collect(self, step_data: StepData) -> None:
        """Collects `step_data` in a thread-safe manner."""
        with self._lock:
            self._buffer.add(step_data)

    @property
    def new_data_buffer(self) -> BufferType:
        """Returns renewed data buffer object."""
        return self._buffer.new()

    def renew(self) -> None:
        """Renews the internal data buffer in a thread-safe manner."""
        with self._lock:
            self._buffer = self.new_data_buffer

    def move_data(self) -> BufferType:
        """Move data's pointer to other object."""
        with self._lock:
            return_data = self._buffer
            self.renew()
            return return_data


class ThreadSafeDataUser(Generic[BufferType]):
    """Uses the collected data in training thread."""

    def __init__(self, collector: ThreadSafeDataCollector[BufferType]) -> None:
        """Constructs the data user object."""
        self.collector = collector
        self._lock = threading.RLock()  # For data user is referred from multiple threads.
        self._buffer = collector.new_data_buffer

    def update(self) -> None:
        """Updates the internal data buffer with new buffer from the
        collector."""
        with self._lock:
            buffer = self.collector.move_data()
            self._buffer.concatenate(buffer)

    def get_dataset(self) -> Dataset[Any]:
        """Retrieves the dataset, concatenated with the new data buffer, and
        updates the internal data buffer accordingly."""
        with self._lock:
            self.update()
            return self._buffer.make_dataset()

    def clear(self) -> None:
        """Clears the current data stored in the data collector and from the
        user."""
        with self._lock:
            self._buffer = self._buffer.new()
            self.collector.renew()

    @property
    def buffer(self) -> BufferType:
        """Returns the reference to the internal buffer."""
        return self._buffer
