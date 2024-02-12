import threading

from .base_thread import BaseThread
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

    def start(self) -> None:
        self.logger.info("Starts background thread.")
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def join(self) -> None:
        self._thread.join()
        self.logger.info("Joined background thread.")
