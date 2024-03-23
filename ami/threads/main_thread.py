import threading
from typing import TypeAlias

from .base_thread import BaseThread
from .shared_object_names import SharedObjectNames
from .thread_control import ThreadController
from .thread_types import ThreadTypes
from .web_api_handler import WebApiHandler

AddressType: TypeAlias = tuple[str, int]  # host, port


class MainThread(BaseThread):
    """Implements the main thread functionality within the ami system.

    Shares the thread command handlers with the training and inference
    threads.
    """

    THREAD_TYPE = ThreadTypes.MAIN

    def __init__(self, address: AddressType = ("0.0.0.0", 8391)) -> None:
        super().__init__()

        self._host = address[0]
        self._port = address[1]
        self.thread_controller = ThreadController()
        self.web_api_handler = WebApiHandler(self.thread_controller, self._host, self._port)
        self._handler_thread = threading.Thread(target=self.web_api_handler.run, daemon=True)

        self.share_object(SharedObjectNames.THREAD_COMMAND_HANDLERS, self.thread_controller.handlers)

    def worker(self) -> None:
        self.logger.info("Start main thread.")
        self.thread_controller.activate()

        self.logger.info(f"Serving system command at 'http://{self._host}:{self._port}'")
        self._handler_thread.start()
        try:
            while not self.thread_controller.wait_for_shutdown(1.0):
                pass
        except KeyboardInterrupt:
            self.thread_controller.shutdown()
            self.logger.info("Shutting down by KeyboardInterrupt.")

        self.logger.info("End main thread.")
