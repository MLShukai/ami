import pytest
import torch

from ami.data.step_data import DataKeys, StepData
from ami.data.utils import DataCollectorsDict, DataUsersDict, ThreadSafeDataCollector

from .buffers.test_base_data_buffer import DataBufferImpl


class TestDataCollectorsAndUsersDict:
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

    @pytest.fixture
    def users_dict(self, collectors_dict: DataCollectorsDict):
        return collectors_dict.get_data_users()

    def test_collect(self, collectors_dict: DataCollectorsDict, step_data: StepData) -> None:
        collectors_dict.collect(step_data)

    def test_from_buffers(self, buffer: DataBufferImpl) -> None:
        collectors = DataCollectorsDict.from_data_buffers(b=buffer)
        assert isinstance(collectors["b"], ThreadSafeDataCollector)

    def test_get_data_users(self, collectors_dict: DataCollectorsDict) -> None:
        assert isinstance(collectors_dict.get_data_users(), DataUsersDict)

    def test_save_and_load_state(self, users_dict: DataUsersDict, tmp_path, collector, step_data):
        data_path = tmp_path / "data"

        collector.collect(step_data)

        users_dict.save_state(data_path)
        assert (data_path / "a").exists()

        for user in users_dict.values():
            user.clear()
            assert len(user.buffer.obs) == 0

        users_dict.load_state(data_path)

        for user in users_dict.values():
            assert len(user.buffer.obs) == 1

    def test_acquire(self, collectors_dict: DataCollectorsDict) -> None:
        """Test acquiring a collector."""
        # Test successful acquire
        collector = collectors_dict.acquire("a")
        assert isinstance(collector, ThreadSafeDataCollector)
        assert "a" in collectors_dict._acquired_collectors

    def test_acquire_non_existent(self, collectors_dict: DataCollectorsDict) -> None:
        """Test acquiring a non-existent collector."""
        with pytest.raises(KeyError, match="Data collector 'non_existent' not found"):
            collectors_dict.acquire("non_existent")

    def test_acquire_already_acquired(self, collectors_dict: DataCollectorsDict) -> None:
        """Test acquiring an already acquired collector."""
        collectors_dict.acquire("a")
        with pytest.raises(KeyError, match="Data collector 'a' is already acquired"):
            collectors_dict.acquire("a")

    def test_collect_with_acquired(self, collectors_dict: DataCollectorsDict, step_data: StepData) -> None:
        """Test collect operation when collectors are acquired."""
        collectors_dict.acquire("a")
        with pytest.raises(RuntimeError, match="Cannot collect while collectors are acquired"):
            collectors_dict.collect(step_data)
