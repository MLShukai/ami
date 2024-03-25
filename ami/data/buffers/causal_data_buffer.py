from collections import deque
from typing import Self

import torch
from torch.utils.data import TensorDataset

from ..step_data import DataKeys, StepData
from .base_data_buffer import BaseDataBuffer


class CausalDataBuffer(BaseDataBuffer):
    """A data buffer which preserve data order."""

    def init(self, max_len: int, key_list: list[DataKeys]) -> None:
        """Initializes data buffer.

        Args:
            max_len: max length of buffer.
            key_list: a list of keys to save whose values to buffer.
        """
        self.__max_len = max_len
        self.__current_len = 0
        self._key_list = key_list
        self.__buffer_dict: dict[DataKeys, deque[torch.Tensor]] = dict()
        for key in key_list:
            self.__buffer_dict[key] = deque(maxlen=max_len)

    def __len__(self) -> int:
        """Returns current data length.

        Returns:
            int: current data length.
        """
        return self.__current_len

    def add(self, step_data: StepData) -> None:
        """Add a single step of data.

        Args:
            step_data: A single step of data.
        """
        for key in self._key_list:
            self.__buffer_dict[key].append(torch.Tensor(step_data[key]))
        if self.__current_len < self.__max_len:
            self.__current_len += 1

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
        self.__current_len = min(len(self) + len(new_data), self.__max_len)

    def make_dataset(self) -> TensorDataset:
        """Make a TensorDataset from current buffer.

        Returns:
            TensorDataset: a TensorDataset created from current buffer.
        """
        tensor_list = []
        for key in self._key_list:
            tensor_list.append(torch.stack(list(self.__buffer_dict[key])))
        return TensorDataset(*tensor_list)
