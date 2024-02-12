import threading

from .base_thread import BaseThread
from .shared_object_pool import SharedObjectNames
from .thread_control import ThreadCommandHandler
from .thread_types import ThreadTypes


class BackgroundThread(BaseThread):
    """Base class for all background thread objects.

    The `THREAD_TYPE` attribute must not be set to `ThreadType.MAIN`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._thread = threading.Thread(target=self.run)

        if self.THREAD_TYPE is ThreadTypes.MAIN:
            raise ValueError("Background `THREAD_TYPE` must not be MAIN!")

    @property
    def thread_command_handler(self) -> ThreadCommandHandler:
        """Retrieves the thread command handler object shared by the main
        thread."""
        if self._thread_command_handler is None:
            handler = self.get_shared_object(ThreadTypes.MAIN, SharedObjectNames.THREAD_COMMAND_HANDLERS)[
                self.THREAD_TYPE
            ]
            if not isinstance(handler, ThreadCommandHandler):
                raise RuntimeError(f"Shared thread command handler object is invalid: {handler}")
            self._thread_command_handler = handler
        return self._thread_command_handler

    def start(self) -> None:
        self.logger.info("Starts background thread.")
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def join(self) -> None:
        self._thread.join()
        self.logger.info("Joined background thread.")
