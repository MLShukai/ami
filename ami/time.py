"""AMI Time Module.

This module provides a custom implementation of time-related functions with time acceleration and pause/resume features.
It wraps the standard Python time module and allows for consistent time control across the AMI system.

Key features:
- Time acceleration: Adjust the speed of time passage in the system.
- Pause/Resume: Ability to pause and resume the flow of time in the system.
- Thread-safe: All operations are protected by locks for use in multi-threaded environments.
- Compatible API: Provides familiar time functions like sleep, time, perf_counter, and monotonic.
- Type annotations: Utilizes Python 3.10+ type annotations for improved type checking and IDE support.
- Original time functions: Provides access to the original time functions with 'fixed_' prefix.

Usage:
    from ami import time

    # Get current time (affected by time scale and pause)
    current_time = time.time()

    # Get current time (not affected by time scale or pause)
    fixed_current_time = time.fixed_time()

    # Sleep for 1 second (affected by time scale and pause)
    time.sleep(1)

    # Sleep for 1 second (not affected by time scale or pause)
    time.fixed_sleep(1)

    # Set time scale (e.g., 2x speed)
    time.set_time_scale(2.0)

    # Get current time scale
    current_scale = time.get_time_scale()

    # Pause time
    time.pause()

    # Resume time
    time.resume()

Note: This module is designed for use within the AMI system and may not be suitable for general-purpose time management.
"""

from __future__ import annotations

import time as _original_time
from functools import wraps
from threading import RLock
from typing import Callable, Concatenate, ParamSpec, TypeVar, cast

T = TypeVar("T")
P = ParamSpec("P")


def with_lock(method: Callable[Concatenate[TimeController, P], T]) -> Callable[Concatenate[TimeController, P], T]:
    @wraps(method)
    def _impl(self: TimeController, *method_args: P.args, **method_kwargs: P.kwargs) -> T:
        with self._lock:
            return method(self, *method_args, **method_kwargs)

    return cast(Callable[Concatenate[TimeController, P], T], _impl)


class TimeController:
    def __init__(self) -> None:
        self._lock = RLock()
        self._anchor_time = _original_time.time()
        self._anchor_perf_counter = _original_time.perf_counter()
        self._anchor_monotonic = _original_time.monotonic()
        self._scaled_anchor_time = _original_time.time()
        self._scaled_anchor_perf_counter = _original_time.perf_counter()
        self._scaled_anchor_monotonic = _original_time.monotonic()

        self._time_scale = 1.0

    @with_lock
    def _update_anchor_values(self) -> None:
        self._anchor_time = _original_time.time()
        self._anchor_perf_counter = _original_time.perf_counter()
        self._anchor_monotonic = _original_time.monotonic()

    @with_lock
    def _update_scaled_anchor_values(self) -> None:
        self._scaled_anchor_time = self.time()
        self._scaled_anchor_perf_counter = self.perf_counter()
        self._scaled_anchor_monotonic = self.monotonic()

    @with_lock
    def time(self) -> float:
        """Return the current time in seconds since the epoch.

        This function is affected by the current time scale and pause state.

        Returns:
            float: The current time in seconds since the epoch.
        """
        delta = _original_time.time() - self._anchor_time
        return self._scaled_anchor_time + delta * self._time_scale

    @with_lock
    def perf_counter(self) -> float:
        """Return the value (in fractional seconds) of a performance counter.

        This function is affected by the current time scale and pause state.

        Returns:
            float: The current value of the performance counter.
        """
        delta = _original_time.perf_counter() - self._anchor_perf_counter
        return self._scaled_anchor_perf_counter + delta * self._time_scale

    @with_lock
    def monotonic(self) -> float:
        """Return the value (in fractional seconds) of a monotonic time
        counter.

        This function is affected by the current time scale and pause state.

        Returns:
            float: The current value of the monotonic time counter.
        """
        delta = _original_time.monotonic() - self._anchor_monotonic
        return self._scaled_anchor_monotonic + delta * self._time_scale

    @with_lock
    def set_time_scale(self, time_scale: float) -> None:
        """Set the time scale for the AMI system.

        Args:
            time_scale (float): The new time scale. Must be greater than 0.

        Raises:
            AssertionError: If time_scale is not greater than 0.
        """
        assert time_scale > 0, "Time scale must be > 0"
        self._update_scaled_anchor_values()
        self._update_anchor_values()
        self._time_scale = time_scale

    @with_lock
    def get_time_scale(self) -> float:
        """Get the current time scale of the AMI system.

        Returns:
            float: The current time scale.
        """
        return self._time_scale

    def sleep(self, secs: float) -> None:
        """Suspend execution for the given number of seconds.

        This function is affected by the current time scale and pause state.

        Args:
            secs (float): The number of seconds to sleep.
        """

        with self._lock:
            time_scale = self._time_scale
        _original_time.sleep(secs / time_scale)


# Create a global instance of TimeController
_time_controller = TimeController()

# Expose the public methods
sleep = _time_controller.sleep
time = _time_controller.time
perf_counter = _time_controller.perf_counter
monotonic = _time_controller.monotonic
set_time_scale = _time_controller.set_time_scale
get_time_scale = _time_controller.get_time_scale

# Expose the original time functions.
fixed_sleep = _original_time.sleep
fixed_time = _original_time.time
