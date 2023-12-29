"""This file contains the abstract base environment and implemented environment
classes."""
from abc import ABC, abstractmethod
from typing import Any

from .actuators import BaseActuator
from .sensors import BaseSensor


class BaseEnvironment(ABC):
    """Abstract base environment class for interacting with real
    environment."""

    @abstractmethod
    def observe(self) -> Any:
        """Observes the data from real environment.

        Returns:
            observation: Read data from real environment.
        """

    @abstractmethod
    def affect(self, action: Any) -> None:
        """Affects the action to the real environment.

        Args:
            action: The action for the real environment.
        """

    def setup(self) -> None:
        """Called at the start of interaction with the agent."""
        pass

    def teardown(self) -> None:
        """Called at the end of interaction with the agent."""
        pass


class SensorActuatorEnv(BaseEnvironment):
    """The environment class that has sensor and actuator."""

    def __init__(self, sensor: BaseSensor, actuator: BaseActuator) -> None:
        self.sensor = sensor
        self.actuator = actuator

    def observe(self) -> Any:
        return self.sensor.read()

    def affect(self, action: Any) -> None:
        self.actuator.operate(action)

    def setup(self) -> None:
        self.sensor.setup()
        self.actuator.setup()

    def teardown(self) -> None:
        self.sensor.teardown()
        self.actuator.teardown()
