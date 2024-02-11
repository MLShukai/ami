import copy
from typing import Any, Callable, Generic

from .._types import ActType, ObsType
from .base_environment import BaseEnvironment


def empty_action_checker(action: Any) -> None:
    pass


class DummyEnvironment(BaseEnvironment[ObsType, ActType], Generic[ObsType, ActType]):
    """Dummy implementation of the environment class."""

    def __init__(
        self,
        observation_generator: Callable[[], ObsType],
        action_checker: Callable[[ActType], None] | None = None,
    ) -> None:
        """Constructs the class.

        Args:
            observation_generator: The callable which returns the observation.
            action_checker: You can set the checker function whether the action is valid or not.
        """
        super().__init__()
        self.observation_generator = observation_generator

        if action_checker is None:
            action_checker = empty_action_checker
        self.action_checker = action_checker

    def observe(self) -> ObsType:
        return self.observation_generator()

    def affect(self, action: ActType) -> None:
        self.action_checker(action)


class SameObservationGenerator(Generic[ObsType]):
    """Generates a consistent observation on each call."""

    def __init__(self, observation: ObsType, deep_copy: bool = True) -> None:
        """Initializes the generator with a fixed observation.

        Args:
            observation: The observation to return on each call.
            deep_copy: Whether to return a deep copy of the observation on each call.
        """
        self.observation = observation
        self.deep_copy = deep_copy

    def __call__(self) -> ObsType:
        """Returns the observation.

        Returns:
            A (deep) copy of the initial observation if `deep_copy` is True, else returns the original observation.
        """
        if self.deep_copy:
            return copy.deepcopy(self.observation)
        return self.observation


class ActionTypeChecker(Generic[ActType]):
    """Validates the type of action provided."""

    def __init__(self, action_type: type[ActType]) -> None:
        """Constructs the action type checker.

        Args:
            action_type: The expected type of the action.
        """
        self.action_type = action_type

    def __call__(self, action: ActType) -> None:
        """Checks if the given action is of the expected type.

        Raises:
            ValueError: If the action is not of the expected type.
        """
        if not isinstance(action, self.action_type):
            raise ValueError(f"Unexpected action type: {type(action)}, expected: {self.action_type}")
