import time

import pytest

from ami.interactions.fixed_interval_interaction import FixedIntervalInteraction
from ami.interactions.interval_adjustors import SleepIntervalAdjustor

from ..helpers import skip_if_platform_is_not_linux


class TestFixedIntervalInteraction:
    @skip_if_platform_is_not_linux()
    @pytest.mark.parametrize("interval", [0.1])
    def test_intervals(self, mock_env, mock_agent, interval: float):
        interaction = FixedIntervalInteraction(
            mock_env,
            mock_agent,
            SleepIntervalAdjustor(interval),
        )

        interaction.setup()
        start = time.perf_counter()
        interaction.step()
        delta_time = time.perf_counter() - start

        assert delta_time == pytest.approx(interval, abs=0.01)
