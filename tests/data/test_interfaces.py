import pickle
from pathlib import Path

import pytest
import torch

from ami.data.interfaces import ThreadSafeDataCollector, ThreadSafeDataUser
from ami.data.step_data import DataKeys, StepData

from .buffers.test_base_data_buffer import DataBufferImpl


class TestDataCollectorAndUser:
    @pytest.fixture
    def collector(self) -> ThreadSafeDataCollector[DataBufferImpl]:
        return ThreadSafeDataCollector(DataBufferImpl.reconstructable_init())

    @pytest.fixture
    def user(self, collector: ThreadSafeDataCollector[DataBufferImpl]) -> ThreadSafeDataUser[DataBufferImpl]:
        return ThreadSafeDataUser(collector)

    @pytest.fixture
    def step_data(self) -> StepData:
        return StepData({DataKeys.OBSERVATION: torch.randn(10)})

    # --- Testing Collector ---
    def test_collect(self, collector: ThreadSafeDataCollector[DataBufferImpl], step_data: StepData) -> None:

        collector.collect(step_data)

    def test_move_data(self, collector: ThreadSafeDataCollector[DataBufferImpl]) -> None:

        buffer = collector._buffer
        moved_buffer = collector.move_data()
        assert buffer is moved_buffer
        assert buffer is not collector._buffer

    # --- Testing User ---
    def test_update(
        self,
        collector: ThreadSafeDataCollector[DataBufferImpl],
        user: ThreadSafeDataUser[DataBufferImpl],
        step_data: StepData,
    ) -> None:
        assert len(user.buffer.obs) == 0
        collector.collect(step_data)
        user.update()
        assert len(user.buffer.obs) == 1

    def test_get_dataset(
        self,
        collector: ThreadSafeDataCollector[DataBufferImpl],
        user: ThreadSafeDataUser[DataBufferImpl],
        step_data: StepData,
    ) -> None:
        collector.collect(step_data)

        dataset = user.get_dataset()
        assert torch.equal(dataset[:][0], step_data[DataKeys.OBSERVATION].unsqueeze(0))

        collector.collect(step_data)
        dataset = user.get_dataset()
        assert torch.equal(dataset[:][0], torch.stack([step_data[DataKeys.OBSERVATION]] * 2))

    def test_clear(
        self,
        collector: ThreadSafeDataCollector[DataBufferImpl],
        user: ThreadSafeDataUser[DataBufferImpl],
        step_data: StepData,
    ) -> None:
        collector.collect(step_data)

        collector_buffer = collector._buffer
        user_buffer = user._buffer
        user.clear()

        assert collector_buffer is not collector._buffer
        assert user_buffer is not user._buffer

    def test_save_state(
        self,
        collector: ThreadSafeDataCollector[DataBufferImpl],
        user: ThreadSafeDataUser[DataBufferImpl],
        step_data: StepData,
        tmp_path: Path,
    ) -> None:
        collector.collect(step_data)

        data_path = tmp_path / "data"
        user.save_state(data_path)

        with open(data_path / "obs.pkl", "rb") as f:
            for i, obs in enumerate(user.buffer.obs):
                assert torch.equal(pickle.load(f)[i], obs)
