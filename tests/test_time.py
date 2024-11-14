import time as original_time

import pytest

from ami import time as ami_time


@pytest.fixture
def reset_time():
    """Reset the time controller before and after each test."""
    ami_time._time_controller = ami_time.TimeController()
    yield
    ami_time._time_controller = ami_time.TimeController()


def test_default_time_scale(reset_time):
    """Verify the default time scale is 1.0."""
    assert ami_time.get_time_scale() == 1.0


def test_set_get_time_scale(reset_time):
    """Verify time scale can be set and retrieved."""
    test_scales = [0.5, 1.0, 2.0, 10.0]
    for scale in test_scales:
        ami_time.set_time_scale(scale)
        assert ami_time.get_time_scale() == scale


def test_set_invalid_time_scale(reset_time):
    """Verify setting invalid time scales raises an error."""
    invalid_scales = [-1.0, 0.0]
    for scale in invalid_scales:
        with pytest.raises(AssertionError):
            ami_time.set_time_scale(scale)


def test_time_monotonicity(reset_time):
    """Verify time values are strictly increasing."""
    times = [ami_time.time() for _ in range(5)]
    assert all(t2 >= t1 for t1, t2 in zip(times, times[1:]))


def test_perf_counter_monotonicity(reset_time):
    """Verify perf_counter values are strictly increasing."""
    times = [ami_time.perf_counter() for _ in range(5)]
    assert all(t2 >= t1 for t1, t2 in zip(times, times[1:]))


def test_monotonic_monotonicity(reset_time):
    """Verify monotonic values are strictly increasing."""
    times = [ami_time.monotonic() for _ in range(5)]
    assert all(t2 >= t1 for t1, t2 in zip(times, times[1:]))


def test_sleep_duration(reset_time):
    """Verify sleep duration at normal speed."""
    ami_time.set_time_scale(1.0)
    sleep_time = 0.1
    start = original_time.time()
    ami_time.sleep(sleep_time)
    elapsed = original_time.time() - start
    assert elapsed == pytest.approx(0.1, abs=0.01)


def test_sleep_with_time_scale(reset_time):
    """Verify sleep duration is affected by time scale."""
    ami_time.set_time_scale(2.0)
    sleep_time = 0.1
    start = original_time.time()
    ami_time.sleep(sleep_time)
    elapsed = original_time.time() - start
    assert elapsed == pytest.approx(0.05, abs=0.01)  # Should take half the time at 2x speed


def test_fixed_sleep_duration(reset_time):
    """Verify fixed_sleep is not affected by time scale."""
    ami_time.set_time_scale(2.0)
    sleep_time = 0.1
    start = original_time.time()
    ami_time.fixed_sleep(sleep_time)
    elapsed = original_time.time() - start
    assert elapsed == pytest.approx(0.1, abs=0.01)  # Normal duration regardless of time scale


def test_fixed_time_vs_scaled_time(reset_time):
    """Verify fixed_time is not affected by time scale."""
    ami_time.set_time_scale(2.0)
    start_time = ami_time.fixed_time()
    start_scaled = ami_time.time()
    original_time.sleep(0.1)

    delta_fixed = ami_time.fixed_time() - start_time
    delta_scaled = ami_time.time() - start_scaled

    # Scaled time should advance twice as fast
    assert delta_scaled == pytest.approx(delta_fixed * 2, abs=0.001)


def test_time_functions_scale_properly(reset_time):
    """Verify all time functions respect time scale."""
    scale = 2.0
    ami_time.set_time_scale(scale)

    # Test each time function
    funcs = [ami_time.time, ami_time.perf_counter, ami_time.monotonic]
    for func in funcs:
        fixed_start = ami_time.fixed_time()
        start = func()
        original_time.sleep(0.1)
        elapsed = func() - start
        fixed_elapsed = ami_time.fixed_time() - fixed_start

        # Scaled elapsed time should be approximately scale times the fixed elapsed time
        assert elapsed == pytest.approx(fixed_elapsed * scale, abs=0.001)
