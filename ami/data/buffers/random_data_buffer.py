from typing import Any, Self

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ..step_data import DataKeys, StepData
from .base_data_buffer import BaseDataBuffer


class RandomDataBuffer(BaseDataBuffer):
    def init(self, max_len: int, key_list: list[DataKeys]) -> None:
        self.__max_len = max_len
        self.__current_len = 0
        self.__key_list = key_list
        self.__buffer_dict: dict[DataKeys, list[Any]] = dict()
        for key in key_list:
            self.__buffer_dict[key] = []

    def __len__(self) -> int:
        return self.__current_len

    def add(self, step_data: StepData) -> None:
        if self.__current_len < self.__max_len:
            for key in self.__key_list:
                self.__buffer_dict[key].append(step_data[key])
            self.__current_len += 1
        else:
            replace_index = np.random.randint(0, self.__max_len)
            for key in self.__key_list:
                self.__buffer_dict[key][replace_index] = step_data[key]

    @property
    def buffer_dict(self) -> dict[DataKeys, list[Any]]:
        return self.__buffer_dict

    def concatenate(self, new_data: Self) -> None:
        for i in range(len(new_data)):
            step_data = StepData()
            for key in self.__key_list:
                step_data[key] = new_data.buffer_dict[key][i]
            self.add(step_data)

    def make_dataset(self) -> TensorDataset:
        tensor_list = []
        for key in self.__key_list:
            tensor_list.append(torch.stack(self.__buffer_dict[key]))
        return TensorDataset(*tensor_list)
