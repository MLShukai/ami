"""This file contains the abstract base agent class."""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .._types import ActType, ObsType


class BaseAgent(ABC, Generic[ObsType, ActType]):
    """Abstract base agent class for communicating with the environment."""

    @abstractmethod
    def step(self, observation: ObsType) -> ActType:
        """Processes the observation and returns an action to the environment.

        Args:
            observation: Data read from the environment.

        Returns:
            action: Action data intended to affect the environment.
        """
        raise NotImplementedError

    def setup(self, observation: ObsType) -> ActType | None:
        """Setup procedure for the Agent.

        Args:
            observation: Initial observation from the environment.

        Returns:
            action: Initial action to be taken in response to the environment during interaction. Returning no action is also an option.
        """
        return None

    def teardown(self, observation: ObsType) -> ActType | None:
        """Teardown procedure for the Agent.

        Args:
            observation: Final observation from the environment.

        Returns:
            action: Final action to be taken in the interaction. Returning no action is also an option.
        """
        return None
