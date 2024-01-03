"""This file contains interface classes and aggregation classes."""
import threading
from typing import Any, Self

from torch.utils.data import Dataset

from .data_buffers import BaseDataBuffer, StepData


class DataCollector:
    """Collects the data in inference thread."""

    def __init__(self, buffer: BaseDataBuffer) -> None:
        """Constructs data collector class."""
        self._buffer = buffer
        self._lock = threading.RLock()

    def collect(self, step_data: StepData) -> None:
        """Collects `step_data` in a thread-safe manner."""
        with self._lock:
            self._buffer.add(step_data)

    @property
    def new_data_buffer(self) -> BaseDataBuffer:
        """Returns renewed data buffer object."""
        return self._buffer.new()

    def renew(self) -> None:
        """Renews the internal data buffer in a thread-safe manner."""
        with self._lock:
            self._buffer = self.new_data_buffer

    def move_data(self) -> BaseDataBuffer:
        """Move data's pointer to other object."""
        with self._lock:
            return_data = self._buffer
            self.renew()
            return return_data


class DataUser:
    """Uses the collected data in training thread."""

    def __init__(self, collector: DataCollector) -> None:
        """Constructs the data user object."""
        self.collector = collector
        self._lock = threading.RLock()  # For data user is referred from multiple threads.
        self._buffer = collector.new_data_buffer

    def get_dataset(self) -> Dataset[Any]:
        """Retrieves the dataset from the current data buffer."""
        with self._lock:
            return self._buffer.make_dataset()

    def get_new_dataset(self) -> Dataset[Any]:
        """Retrieves the dataset, concatenated with the new data buffer, and
        updates the internal data buffer accordingly."""
        with self._lock:
            buffer = self.collector.move_data()
            self._buffer.concatenate(buffer)

            return self.get_dataset()

    def clear(self) -> None:
        """Clears the current data stored in the data collector and from the
        user."""
        with self._lock:
            self._buffer = self._buffer.new()
            self.collector.renew()


class DataUsersAggregation(dict[str, DataUser]):
    """A class for aggregating `DataUsers` to share them from the inference
    thread to the training thread."""


class DataCollectorsAggregation(dict[str, DataCollector]):
    """A class for aggregating `DataCollectors` to invoke their `collect`
    methods within the agent class."""

    def collect(self, step_data: StepData) -> None:
        """Calls the `collect` method on every `DataCollector` item."""
        for v in self.values():
            v.collect(step_data)

    @classmethod
    def from_data_buffers(cls, **data_buffers: BaseDataBuffer) -> Self:
        """Constructs the class from data buffers.

        Args:
            **data_buffers: Key-value pairs of names and corresponding implemented DataBuffer objects.
        """
        return cls({k: DataCollector(v) for k, v in data_buffers.items()})

    def get_data_users(self) -> DataUsersAggregation:
        """Creates a `DataUsersAggregation` from the item's value."""
        return DataUsersAggregation({k: DataUser(v) for k, v in self.items()})
