import time

import pytest

from ami.checkpointing.checkpoint_schedulers import FixedTimeIntervalScheduler


class TestFixedTimeIntervalScheduler:
    @pytest.mark.parametrize("interval", [0.001, 0.01])
    def test_is_available(self, interval: float):
        scheduler = FixedTimeIntervalScheduler(interval)

        assert scheduler.is_available()
        assert not scheduler.is_available()
        time.sleep(interval)
        assert scheduler.is_available()
        assert not scheduler.is_available()
