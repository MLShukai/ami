"""This file contains the abstract base sensor and sensor wrapper class."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic

from typing_extensions import override

from ..._types import ObsType, WrapperObsType


class BaseSensor(ABC, Generic[ObsType]):
    """Abstract base sensor class for observing data from the real
    environment."""

    @abstractmethod
    def read(self) -> ObsType:
        """Read and return data from the actual sensor.

        Returns:
            observation: Read data from the actuatual sensor.
        """
        raise NotImplementedError

    def setup(self) -> None:
        """Called at the start of interaction with the agent."""
        pass

    def teardown(self) -> None:
        """Called at the end of interaction with the agent."""
        pass

    def save_state(self, path: Path) -> None:
        """Saves the internal state to the `path`."""
        pass

    def load_state(self, path: Path) -> None:
        """Loads the internal state from the `path`."""
        pass


class BaseSensorWrapper(BaseSensor[WrapperObsType], Generic[WrapperObsType, ObsType]):
    """Wraps the sensor class for modifying the observation.

    You must override :meth:`wrap_observation` method.
    """

    def __init__(self, sensor: BaseSensor[ObsType], *args: Any, **kwds: Any) -> None:
        """Constructs the wrapper class.

        Args:
            sensor: Instance of the sensor that will be wrapped.
        """
        self._sensor = sensor

    @abstractmethod
    def wrap_observation(self, observation: ObsType) -> WrapperObsType:
        """Wraps the observation and return it.

        Args:
            observation: The observation read from the wrapped sensor.

        Returns:
            observation: The modified observation.
        """
        raise NotImplementedError

    def read(self) -> WrapperObsType:
        return self.wrap_observation(self._sensor.read())

    def setup(self) -> None:
        return self._sensor.setup()

    def teardown(self) -> None:
        return self._sensor.teardown()

    @override
    def save_state(self, path: Path) -> None:
        return self._sensor.save_state(path)

    @override
    def load_state(self, path: Path) -> None:
        return self._sensor.load_state(path)
