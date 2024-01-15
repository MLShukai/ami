"""This file contains dictionary classes used for Hydra instantiation
utilities."""
from typing import Self

from .data_buffers import BaseDataBuffer, StepData
from .interfaces import DataCollector, DataUser


class DataUsersDict(dict[str, DataUser]):
    """A class for aggregating `DataUsers` to share them from the inference
    thread to the training thread."""


class DataCollectorsDict(dict[str, DataCollector]):
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

    def get_data_users(self) -> DataUsersDict:
        """Creates a `DataUsersDict` from the item's value."""
        return DataUsersDict({k: DataUser(v) for k, v in self.items()})
