import pickle
import time
from collections import deque
from pathlib import Path

import torch
from torch.utils.data import TensorDataset
from typing_extensions import Self, override

from ..step_data import DataKeys, StepData
from .base_data_buffer import BaseDataBuffer


class CausalDataBuffer(BaseDataBuffer):
    """A data buffer which preserve data order."""

    def __init__(self, max_len: int, key_list: list[DataKeys | str]) -> None:
        """Initializes data buffer.

        Args:
            max_len: max length of buffer.
            key_list: a list of keys to save whose values to buffer.
        """
        assert len(key_list) > 0, "`key_list` must have at least one element!"

        self.__max_len = max_len
        self._key_list = [DataKeys(key) for key in key_list]
        self.__buffer_dict: dict[DataKeys, deque[torch.Tensor]] = dict()
        for key in self._key_list:
            self.__buffer_dict[key] = deque(maxlen=max_len)

        self._added_times: deque[float] = deque(maxlen=max_len)

    def __len__(self) -> int:
        """Returns current data length.

        Returns:
            int: current data length.
        """
        return len(self.__buffer_dict[self._key_list[0]])

    def add(self, step_data: StepData) -> None:
        """Add a single step of data.

        Args:
            step_data: A single step of data.
        """
        for key in self._key_list:
            self.__buffer_dict[key].append(torch.Tensor(step_data[key]).cpu())

        self._added_times.append(time.time())

    @property
    def buffer_dict(self) -> dict[DataKeys, deque[torch.Tensor]]:
        return self.__buffer_dict

    def concatenate(self, new_data: Self) -> None:
        """Concatenates another buffer to this buffer.

        Args:
            new_data: A buffer to concatenate.
        """
        for key in self._key_list:
            self.buffer_dict[key] += new_data.buffer_dict[key]
        self._added_times += new_data._added_times

    def make_dataset(self) -> TensorDataset:
        """Make a TensorDataset from current buffer.

        Returns:
            TensorDataset: a TensorDataset created from current buffer.
        """
        tensor_list = []
        for key in self._key_list:
            tensor_list.append(torch.stack(list(self.__buffer_dict[key])))
        return TensorDataset(*tensor_list)

    def count_data_added_since(self, previous_get_time: float) -> int:
        """Counts the number of data points added since a specified time.

        Args:
            previous_get_time: The reference time to count from.

        Returns:
            int: The number of data points added since the specified time.
        """
        for i, t in enumerate(reversed(self._added_times)):
            if t < previous_get_time:
                return i
        return len(self._added_times)

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        for key, value in self.__buffer_dict.items():
            file_name = path / (key + ".pkl")
            with open(file_name, "wb") as f:
                pickle.dump(value, f)

        with open(path / "_added_times.pkl", "wb") as f:
            pickle.dump(self._added_times, f)

    @override
    def load_state(self, path: Path) -> None:
        for key in self.__buffer_dict.keys():
            file_name = path / (key + ".pkl")
            with open(file_name, "rb") as f:
                self.__buffer_dict[key] = deque(pickle.load(f), maxlen=self.__max_len)

        with open(path / "_added_times.pkl", "rb") as f:
            self._added_times = deque(pickle.load(f), maxlen=self.__max_len)
