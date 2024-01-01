import pytest
import torch

from ami.data import DataKeys, StepData
from ami.data.utils import DataCollector, DataUser

from .test_data_pools import DataPoolImpl


class TestDataCollectorAndUser:
    @pytest.fixture
    def collector(self) -> DataCollector:
        return DataCollector(DataPoolImpl())

    @pytest.fixture
    def user(self, collector: DataCollector) -> DataUser:
        return collector.data_user

    @pytest.fixture
    def step_data(self) -> StepData:
        return StepData({DataKeys.OBSERVATION: torch.randn(10)})

    # --- Testing Collector ---
    def test_collect(self, collector: DataCollector, step_data: StepData) -> None:

        collector.collect(step_data)

    def test_move_data(self, collector: DataCollector) -> None:

        pool = collector._pool
        moved_pool = collector.move_data()
        assert pool is moved_pool
        assert pool is not collector._pool

    def test_data_user(self, collector: DataCollector) -> None:
        assert isinstance(collector.data_user, DataUser)

    # --- Testing User ---
    def test_get_new_dataset(self, collector: DataCollector, user: DataUser, step_data: StepData) -> None:
        collector.collect(step_data)

        dataset = user.get_new_dataset()
        assert torch.equal(dataset[:][0], step_data[DataKeys.OBSERVATION].unsqueeze(0))

        collector.collect(step_data)
        dataset = user.get_new_dataset()
        assert torch.equal(dataset[:][0], torch.stack([step_data[DataKeys.OBSERVATION]] * 2))

    def test_clear(self, collector: DataCollector, user: DataUser, step_data: StepData) -> None:
        collector.collect(step_data)

        collector_pool = collector._pool
        user_pool = user._pool
        user.clear()

        assert collector_pool is not collector._pool
        assert user_pool is not user._pool
