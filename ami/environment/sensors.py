"""This file contains the abstract base sensor, sensor wrappers, and
implemented sensor classes."""
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
