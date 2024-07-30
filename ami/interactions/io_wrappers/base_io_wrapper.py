from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ami.checkpointing import SaveAndLoadStateMixin
from ami.threads.thread_control import PauseResumeEventMixin

WrappingType = TypeVar("WrappingType")
WrappedType = TypeVar("WrappedType")


class BaseIOWrapper(ABC, Generic[WrappingType, WrappedType], PauseResumeEventMixin, SaveAndLoadStateMixin):
    """Base wrapper class for observation/action wrapper class in
    Interaction."""

    @abstractmethod
    def wrap(self, input: WrappingType) -> WrappedType:
        """Wraps input data.

        Args:
            input: The data to be wrapped.

        Returns:
            WrappedType: The wrapped data.
        """
        raise NotImplementedError

    def setup(self) -> None:
        """Called at the start of the interaction."""
        pass

    def teardown(self) -> None:
        """Called at the end of the interaction."""
        pass


# Alias
BaseObservationWrapper = BaseIOWrapper
BaseActionWrapper = BaseIOWrapper
