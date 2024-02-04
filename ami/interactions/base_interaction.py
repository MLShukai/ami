from abc import ABC, abstractmethod


class BaseInteraction(ABC):
    """The base class for all interaction classes."""

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
