"""This file contains interface classes and aggregation classes."""
from __future__ import annotations

import threading

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

    @property
    def data_user(self) -> DataUser:
        """To user object, used in the training thread."""
        return DataUser(self)


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
