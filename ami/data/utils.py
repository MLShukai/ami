"""This file contains dictionary classes used for Hydra instantiation
utilities.

Assumed usage:

    ```yaml:config.yaml
    _target_: ami.data.utils.DataCollectorsDict.from_data_buffers

    some_buffer:
        _target_: <path.to.SomeBuffer>
        some_arg: ...
    ```

    >>> import hydra
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.load("config.yaml")
    >>> collector = hydra.utils.instantiate(cfg)
"""
from collections import UserDict
from pathlib import Path
from typing import Any

from typing_extensions import Self, override

from ami.checkpointing import SaveAndLoadStateMixin

from .buffers.base_data_buffer import BaseDataBuffer
from .interfaces import ThreadSafeDataCollector, ThreadSafeDataUser
from .step_data import StepData


class DataUsersDict(UserDict[str, ThreadSafeDataUser[Any]], SaveAndLoadStateMixin):
    """A class for aggregating `DataUsers` to share them from the inference
    thread to the training thread."""

    @override
    def save_state(self, path: Path) -> None:
        """Saves the internal data buffer."""
        path.mkdir()
        for name, user in self.items():
            user.save_state(path / name)

    @override
    def load_state(self, path: Path) -> None:
        """Loads the internal buffer state from `path`."""
        for name, user in self.items():
            user.load_state(path / name)


class DataCollectorsDict(UserDict[str, ThreadSafeDataCollector[Any]]):
    """A class for aggregating `DataCollectors` to invoke their `collect`
    methods within the agent class.

    Once a collector is acquired, it cannot be released until system
    restart. Acquired collectors are excluded from the collect
    operation.
    """

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self._acquired_collectors: set[str] = set()

    def acquire(self, collector_name: str) -> ThreadSafeDataCollector[Any]:
        """Acquire permanent exclusive access to a specific data collector.

        Once acquired, a collector cannot be released until system restart.

        Args:
            collector_name: Name of the collector to acquire

        Returns:
            ThreadSafeDataCollector[Any]: The acquired data collector

        Raises:
            KeyError: If collector_name is already acquired or doesn't exist
        """
        if collector_name in self._acquired_collectors:
            raise KeyError(f"Data collector '{collector_name}' is already acquired.")
        if collector_name not in self:
            raise KeyError(f"Data collector '{collector_name}' not found.")

        self._acquired_collectors.add(collector_name)
        return self[collector_name]

    def collect(self, step_data: StepData) -> None:
        """Calls the `collect` method on every non-acquired `DataCollector`
        item.

        Args:
            step_data: Data to be collected

        Raises:
            RuntimeError: If any collectors are currently acquired
        """
        if len(self._acquired_collectors) > 0:
            acquired_list = ", ".join(sorted(self._acquired_collectors))
            raise RuntimeError(f"Cannot collect while collectors are acquired: {acquired_list}")
        for v in self.values():
            v.collect(step_data)

    @classmethod
    def from_data_buffers(cls, **data_buffers: BaseDataBuffer) -> Self:
        """Constructs the class from data buffers.

        Args:
            **data_buffers: Key-value pairs of names and corresponding implemented DataBuffer objects.
        """
        return cls({k: ThreadSafeDataCollector(v) for k, v in data_buffers.items()})

    def get_data_users(self) -> DataUsersDict:
        """Creates a `DataUsersDict` from the item's value."""
        return DataUsersDict({k: ThreadSafeDataUser(v) for k, v in self.items()})
