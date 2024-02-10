import threading

from .base_thread import BaseThread
from .shared_object_pool import SharedObjectNames
from .thread_control import ThreadCommandHandler
from .thread_types import ThreadTypes


class BackgroundThread(BaseThread):
    _thread_command_handler: ThreadCommandHandler | None = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls.THREAD_TYPE is ThreadTypes.MAIN:
            raise RuntimeError("Background `THREAD_TYPE` must not be MAIN!")

    def __init__(self) -> None:
        super().__init__()
        self._thread = threading.Thread(target=self.run)

    @property
    def thread_command_handler(self) -> ThreadCommandHandler:
        if self._thread_command_handler is None:
            handler = self.get_shared_object(ThreadTypes.MAIN, SharedObjectNames.THREAD_COMMAND_HANDLERS)[
                self.THREAD_TYPE
            ]
            if isinstance(handler, ThreadCommandHandler):
                self._thread_command_handler = handler
            else:
                raise RuntimeError(f"Shared thread command handler object is invalid: {handler}")
        return self._thread_command_handler

    def start(self) -> None:
        self.logger.info("Starts background thread.")
        self._thread.start()

    def join(self) -> None:
        self._thread.join()
        self.logger.info("Joined background thread.")
