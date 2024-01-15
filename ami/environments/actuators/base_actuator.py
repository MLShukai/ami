"""This file contains the abstract base actuator and actuator wrappers
class."""
from abc import ABC, abstractmethod
from typing import Any


class BaseActuator(ABC):
    """Abstract base actuator class for affecting actiion to the real
    environment."""

    @abstractmethod
    def operate(self, action: Any) -> None:
        """Operates the real actuator by `action`.

        Args:
            action: The command for actuator.
        """
        raise NotImplementedError

    def setup(self) -> None:
        """Called at the start of interaction with the agent."""
        pass

    def teardown(self) -> None:
        """Called at the end of interaction with the agent."""
        pass


class BaseActuatorWrapper(BaseActuator):
    """Wraps the actuator class for modifying the action.

    You must override :meth:`wrap_action` method for wrapping action.
    """

    def __init__(self, actuator: BaseActuator, *args: Any, **kwds: Any) -> None:
        """Constructs the wrapper class.

        Args:
            actuator: Instance of actuator that will be wrapped.
        """
        self._actuator = actuator

    @abstractmethod
    def wrap_action(self, action: Any) -> Any:
        """Wraps the action and return it.

        Args:
            action: The action from the agent, controller, etc.

        Returns:
            action: The modified action.
        """
        raise NotImplementedError

    def operate(self, action: Any) -> None:
        return self._actuator.operate(self.wrap_action(action))

    def setup(self) -> None:
        return self._actuator.setup()

    def teardown(self) -> None:
        return self._actuator.teardown()
