class SystemEventCallbackMixin:
    """Mixin class for implementing the class which has system event callbacks."""

    def on_paused(self) -> None:
        """Callback function called when the system is paused."""
        pass

    def on_resumed(self) -> None:
        """Callback function called when the system is resumed."""
        pass
