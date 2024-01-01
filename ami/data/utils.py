"""This file contains interface classes and aggregation classes."""
import threading
from typing import Self

from torch.utils.data import Dataset

from .data_pools import BaseDataPool, StepData


class DataCollector:
    """Collects the data in inference thread."""

    def __init__(self, pool: BaseDataPool) -> None:
        """Constructs data collector class."""
        self._pool = pool
        self._lock = threading.RLock()

    def collect(self, step_data: StepData) -> None:
        """Collects `step_data` in a thread-safe manner."""
        with self._lock:
            self._pool.add(step_data)

    @property
    def new_data_pool(self) -> BaseDataPool:
        """Returns renewed data pool object."""
        return self._pool.new()

    def renew(self) -> None:
        """Renews the internal data pool in a thread-safe manner."""
        with self._lock:
            self._pool = self.new_data_pool

    def move_data(self) -> BaseDataPool:
        """Move data's pointer to other object."""
        with self._lock:
            return_data = self._pool
            self.renew()
            return return_data


class DataUser:
    """Uses the collected data in training thread."""

    def __init__(self, collector: DataCollector) -> None:
        """Constructs the data user object."""
        self.collector = collector
        self._lock = threading.RLock()  # For data user is referred from multiple threads.
        self._pool = collector.new_data_pool

    def get_dataset(self) -> Dataset:
        """Retrieves the dataset from the current data pool."""
        with self._lock:
            return self._pool.make_dataset()

    def get_new_dataset(self) -> Dataset:
        """Retrieves the dataset, concatenated with the new data pool, and
        updates the internal data pool accordingly."""
        with self._lock:
            pool = self.collector.move_data()
            self._pool.concatenate(pool)

            return self.get_dataset()

    def clear(self) -> None:
        """Clears the current data stored in the data collector and from the
        user."""
        with self._lock:
            self._pool = self._pool.new()
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
    def from_data_pools(cls, **data_pools: BaseDataPool) -> Self:
        """Constructs the class from data pools.

        Args:
            **data_pools: Key-value pairs of names and corresponding implemented DataPool objects.
        """
        return cls({k: DataCollector(v) for k, v in data_pools.items()})

    def get_data_users(self) -> DataUsersAggregation:
        """Creates a `DataUsersAggregation` from the item's value."""
        return DataUsersAggregation({k: DataUser(v) for k, v in self.items()})
