"""`IntervalAdjustor`は複数回`adjust`コールを行った際の分散と平均を元に、その性能を検証する。"""
import statistics

import pytest

from ami.interactions.interval_adjustors import (
    BaseIntervalAdjustor,
    SleepIntervalAdjustor,
)


def compute_adjustor_spec(adjustor: BaseIntervalAdjustor, num_trial: int) -> tuple[float, float]:
    """`num_trial`の回数だけ`adjust`を実行し，経過時間の平均と標準偏差を返す．"""

    delta_times = []
    adjustor.reset()
    for _ in range(num_trial):
        delta_times.append(adjustor.adjust())

    return statistics.mean(delta_times), statistics.stdev(delta_times)


class TestSleepIntervalAdjustor:
    @pytest.mark.parametrize(
        """interval,num_trial""",
        [
            (0.01, 50),
            (0.05, 10),
        ],
    )
    def test_adjust(self, interval: float, num_trial: int) -> None:
        adjustor = SleepIntervalAdjustor(interval)
        mean, std = compute_adjustor_spec(adjustor, num_trial)
        print(mean, std)
        assert mean == pytest.approx(interval, abs=0.001)  # 許容誤差 0.001秒
        assert std < interval * 0.1
