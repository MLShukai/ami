from abc import ABC, abstractmethod

from ._types import ActType, ObsType
from .agents.base_agent import BaseAgent
from .environments.base_environment import BaseEnvironment


class BaseInteraction(ABC):
    """The base class for all interaction classes."""

    def __init__(self, environment: BaseEnvironment[ObsType, ActType], agent: BaseAgent[ObsType, ActType]) -> None:
        """Initializes the interaction with specified environment and agent."""
        self.environment = environment
        self.agent = agent

    def setup(self) -> None:
        """Called at the start of the interaction."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Executes a single step of interaction.

        This method is called repeatedly by the inference thread.
        """
        pass

    def teardown(self) -> None:
        """Called at the end of the interaction."""
        pass
