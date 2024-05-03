import time

import pytest

from ami.checkpointing.checkpoint_schedulers import (
    Checkpointing,
    FixedTimeIntervalCheckpointScheduler,
)


class TestFixedTimeIntervalCheckpointScheduler:
    @pytest.fixture
    def checkpointing(self, tmp_path):
        return Checkpointing(tmp_path / "checkpoints")

    @pytest.mark.parametrize("interval", [0.001, 0.01])
    def test_is_available(self, interval: float, checkpointing: Checkpointing):
        scheduler = FixedTimeIntervalCheckpointScheduler(checkpointing, interval)

        assert scheduler.is_available()
        assert not scheduler.is_available()
        time.sleep(interval)
        assert scheduler.is_available()
        assert not scheduler.is_available()
