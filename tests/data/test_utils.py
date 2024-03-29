import pytest
import torch

from ami.data.step_data import DataKeys, StepData
from ami.data.utils import DataCollectorsDict, DataUsersDict, ThreadSafeDataCollector

from .buffers.test_base_data_buffer import DataBufferImpl


class TestDataCollectorsDict:
    @pytest.fixture
    def buffer(self) -> DataBufferImpl:
        return DataBufferImpl.reconstructable_init()

    @pytest.fixture
    def collector(self, buffer: DataBufferImpl) -> ThreadSafeDataCollector:
        return ThreadSafeDataCollector(buffer)

    @pytest.fixture
    def step_data(self) -> StepData:
        return StepData({DataKeys.OBSERVATION: torch.randn(10)})

    @pytest.fixture
    def collectors_dict(self, collector: ThreadSafeDataCollector) -> DataCollectorsDict:
        return DataCollectorsDict(a=collector)

    def test_collect(self, collectors_dict: DataCollectorsDict, step_data: StepData) -> None:
        collectors_dict.collect(step_data)

    def test_from_buffers(self, buffer: DataBufferImpl) -> None:
        collectors = DataCollectorsDict.from_data_buffers(b=buffer)
        assert isinstance(collectors["b"], ThreadSafeDataCollector)

    def test_get_data_users(self, collectors_dict: DataCollectorsDict) -> None:
        assert isinstance(collectors_dict.get_data_users(), DataUsersDict)
