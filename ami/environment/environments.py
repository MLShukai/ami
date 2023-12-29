"""This file contains the abstract base environment and implemented environment
classes."""
from abc import ABC, abstractmethod
from typing import Any


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

    def setup(self):
        """Called at the start of interaction with the agent."""
        pass

    def teardown(self):
        """Called at the end of interaction with the agent."""
        pass
