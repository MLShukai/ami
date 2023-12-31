from typing import Self

import pytest
import torch
from torch.utils.data import TensorDataset

from ami.data.data_pools import BaseDataPool, DataKeys, StepData


class TestStepData:
    def test_copy(self):
        sd = StepData()
        sd["a"] = [1, 2, 3]

        copied = sd.copy()
        assert sd is not copied
        assert sd["a"] is not copied["a"]


class DataPoolImpl(BaseDataPool):
    def init(self) -> None:
        self.obs: list[torch.Tensor] = []

    def add(self, step_data: StepData) -> None:
        self.obs.append(step_data[DataKeys.OBSERVATION])

    def concatenate(self, new_data: Self) -> None:
        self.obs += new_data.obs

    def make_dataset(self) -> TensorDataset:
        return TensorDataset(torch.stack(self.obs))


class TestDataPool:
    @pytest.fixture
    def data_pool(self) -> DataPoolImpl:
        return DataPoolImpl()

    @pytest.fixture
    def data_pool_added(self, data_pool: DataPoolImpl) -> DataPoolImpl:
        step_data = StepData({DataKeys.OBSERVATION: torch.randn(10)})
        data_pool.add(step_data)
        return data_pool

    def test_new(self, data_pool_added: DataPoolImpl) -> None:

        new = data_pool_added.new()
        assert new is not data_pool_added
        assert new.obs == []
        assert new.obs != data_pool_added.obs
