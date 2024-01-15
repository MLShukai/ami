"""This file contains the abstract base sensor and sensor wrapper class."""
from abc import ABC, abstractmethod
from typing import Any


class BaseSensor(ABC):
    """Abstract base sensor class for observing data from the real
    environment."""

    @abstractmethod
    def read(self) -> Any:
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


class SensorWrapper(BaseSensor):
    """Wraps the sensor class for modifying the observation.

    You must override :meth:`wrap_observation` method.
    """

    def __init__(self, sensor: BaseSensor, *args: Any, **kwds: Any) -> None:
        """Constructs the wrapper class.

        Args:
            sensor: Instance of the sensor that will be wrapped.
        """
        self._sensor = sensor

    @abstractmethod
    def wrap_observation(self, observation: Any) -> Any:
        """Wraps the observation and return it.

        Args:
            observation: The observation read from the wrapped sensor.

        Returns:
            observation: The modified observation.
        """
        raise NotImplementedError

    def read(self) -> Any:
        return self.wrap_observation(self._sensor.read())

    def setup(self) -> None:
        return self._sensor.setup()

    def teardown(self) -> None:
        return self._sensor.teardown()
