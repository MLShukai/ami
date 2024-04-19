import json
import pickle
from pathlib import Path
from typing import Self

import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing_extensions import override

from ..step_data import DataKeys, StepData
from .base_data_buffer import BaseDataBuffer


class RandomDataBuffer(BaseDataBuffer):
    """A data buffer which does not preserve data order."""

    def __init__(self, max_len: int, key_list: list[DataKeys | str]) -> None:
        """Initializes data buffer.

        Args:
            max_len: max length of buffer.
            key_list: a list of keys to save whose values to buffer.
        """
        assert len(key_list) > 0, "`key_list` must have at least one element!"

        self.__max_len = max_len
        self.__key_list = [DataKeys(key) for key in key_list]
        self.__buffer_dict: dict[DataKeys, list[torch.Tensor]] = dict()
        for key in self.__key_list:
            self.__buffer_dict[key] = []

    def __len__(self) -> int:
        """Returns current data length.

        Returns:
            int: current data length.
        """
        return len(self.__buffer_dict[self.__key_list[0]])

    def add(self, step_data: StepData) -> None:
        """Add a single step of data.

        Args:
            step_data: A single step of data.
        """
        if len(self) < self.__max_len:
            for key in self.__key_list:
                self.__buffer_dict[key].append(torch.Tensor(step_data[key]).cpu())
        else:
            replace_index = np.random.randint(0, self.__max_len)
            for key in self.__key_list:
                self.__buffer_dict[key][replace_index] = step_data[key]

    @property
    def buffer_dict(self) -> dict[DataKeys, list[torch.Tensor]]:
        return self.__buffer_dict

    def concatenate(self, new_data: Self) -> None:
        """Concatenates another buffer to this buffer.

        Args:
            new_data: A buffer to concatenate.
        """
        for i in range(len(new_data)):
            step_data = StepData()
            for key in self.__key_list:
                step_data[key] = new_data.buffer_dict[key][i]
            self.add(step_data)

    def make_dataset(self) -> TensorDataset:
        """Make a TensorDataset from current buffer.

        Returns:
            TensorDataset: a TensorDataset created from current buffer.
        """
        tensor_list = []
        for key in self.__key_list:
            tensor_list.append(torch.stack(self.__buffer_dict[key]))
        return TensorDataset(*tensor_list)

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        for key, value in self.__buffer_dict.items():
            file_name = path / (key + ".pkl")
            with open(file_name, "wb") as f:
                pickle.dump(value, f)

    @override
    def load_state(self, path: Path) -> None:
        for key in self.__buffer_dict.keys():
            file_name = path / (key + ".pkl")
            with open(file_name, "rb") as f:
                self.__buffer_dict[key] = pickle.load(f)[: self.__max_len]
