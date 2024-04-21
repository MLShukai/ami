from pathlib import Path


class SaveAndLoadStateMixin:
    """Mixin class for implementing the class which has save and load state
    methods."""

    def save_state(self, path: Path) -> None:
        """Saves the internal state to the `path`."""
        pass

    def load_state(self, path: Path) -> None:
        """Loads the internal state from the `path`"""
        pass
