"""This file contains the abstract base environment class."""
from abc import ABC, abstractmethod
from typing import Any, Generic

from ami.checkpointing import SaveAndLoadStateMixin
from ami.threads import PauseResumeEventMixin

from .._types import ActType, ObsType


class BaseEnvironment(ABC, Generic[ObsType, ActType], SaveAndLoadStateMixin, PauseResumeEventMixin):
    """Abstract base environment class for interacting with real
    environment."""

    @abstractmethod
    def observe(self) -> ObsType:
        """Observes the data from real environment.

        Returns:
            observation: Read data from real environment.
        """

    @abstractmethod
    def affect(self, action: ActType) -> None:
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
