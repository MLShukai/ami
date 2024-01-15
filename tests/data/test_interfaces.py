import pytest
import torch

from ami.data.interfaces import DataCollector, DataUser
from ami.data.step_data import DataKeys, StepData

from .buffers.test_base_data_buffer import DataBufferImpl


class TestDataCollectorAndUser:
    @pytest.fixture
    def collector(self) -> DataCollector:
        return DataCollector(DataBufferImpl())

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

        buffer = collector._buffer
        moved_buffer = collector.move_data()
        assert buffer is moved_buffer
        assert buffer is not collector._buffer

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

        collector_buffer = collector._buffer
        user_buffer = user._buffer
        user.clear()

        assert collector_buffer is not collector._buffer
        assert user_buffer is not user._buffer
