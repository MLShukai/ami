from typing import Self

import pytest
import torch
from torch.utils.data import TensorDataset

from ami.data.data_buffers import BaseDataBuffer, DataKeys, StepData


class TestStepData:
    def test_copy(self) -> None:
        sd = StepData()
        sd["a"] = [1, 2, 3]

        copied = sd.copy()
        assert sd is not copied
        assert sd["a"] is not copied["a"]


class DataBufferImpl(BaseDataBuffer):
    def init(self) -> None:
        self.obs: list[torch.Tensor] = []

    def add(self, step_data: StepData) -> None:
        self.obs.append(step_data[DataKeys.OBSERVATION])

    def concatenate(self, new_data: Self) -> None:
        self.obs += new_data.obs

    def make_dataset(self) -> TensorDataset:
        return TensorDataset(torch.stack(self.obs))


class TestDataBuffer:
    @pytest.fixture
    def data_buffer(self) -> DataBufferImpl:
        return DataBufferImpl()

    @pytest.fixture
    def data_buffer_added(self, data_buffer: DataBufferImpl) -> DataBufferImpl:
        step_data = StepData({DataKeys.OBSERVATION: torch.randn(10)})
        data_buffer.add(step_data)
        return data_buffer

    def test_new(self, data_buffer_added: DataBufferImpl) -> None:

        new = data_buffer_added.new()
        assert new is not data_buffer_added
        assert new.obs == []
        assert new.obs != data_buffer_added.obs
