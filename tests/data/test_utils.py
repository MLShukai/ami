import pytest
import torch

from ami.data import DataKeys, StepData
from ami.data.utils import (
    DataCollector,
    DataCollectorsAggregation,
    DataUser,
    DataUsersAggregation,
)

from .test_data_pools import DataPoolImpl


class TestDataCollectorAndUser:
    @pytest.fixture
    def collector(self) -> DataCollector:
        return DataCollector(DataPoolImpl())

    @pytest.fixture
    def user(self, collector: DataCollector) -> DataUser:
        return DataUser(collector)

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


class TestDataCollectorsAggregation:
    @pytest.fixture
    def pool(self) -> DataPoolImpl:
        return DataPoolImpl()

    @pytest.fixture
    def collector(self, pool: DataPoolImpl) -> DataCollector:
        return DataCollector(pool)

    @pytest.fixture
    def step_data(self) -> StepData:
        return StepData({DataKeys.OBSERVATION: torch.randn(10)})

    @pytest.fixture
    def collectors_aggregation(self, collector: DataCollector) -> DataCollectorsAggregation:
        return DataCollectorsAggregation(a=collector)

    def test_collect(self, collectors_aggregation: DataCollectorsAggregation, step_data: StepData) -> None:
        collectors_aggregation.collect(step_data)

    def test_from_pools(self, pool: DataPoolImpl) -> None:
        collectors = DataCollectorsAggregation.from_data_pools(b=pool)
        assert isinstance(collectors["b"], DataCollector)

    def test_get_data_users(self, collectors_aggregation: DataCollectorsAggregation) -> None:
        assert isinstance(collectors_aggregation.get_data_users(), DataUsersAggregation)
