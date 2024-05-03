import time
from typing import TypeAlias

from .base_thread import BaseThread
from .shared_object_names import SharedObjectNames
from .thread_control import ThreadController, ThreadControllerStatus
from .thread_types import ThreadTypes
from .web_api_handler import ControlCommands, WebApiHandler

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
        self.web_api_handler = WebApiHandler(ThreadControllerStatus(self.thread_controller), self._host, self._port)

        self.share_object(SharedObjectNames.THREAD_COMMAND_HANDLERS, self.thread_controller.handlers)

    def worker(self) -> None:
        self.logger.info("Start main thread.")
        self.thread_controller.activate()

        self.logger.info(f"Serving system command at 'http://{self._host}:{self._port}'")
        self.web_api_handler.run_in_background()

        try:
            while not self.thread_controller.is_shutdown():

                self.process_received_commands()

                time.sleep(0.001)

        except KeyboardInterrupt:
            self.logger.info("Shutting down by KeyboardInterrupt.")

        finally:
            self.logger.info("Shutting down...")
            self.thread_controller.shutdown()

        self.logger.info("End main thread.")

    def process_received_commands(self) -> None:
        """Processes the received commands from web api handler."""
        while self.web_api_handler.has_commands():
            match self.web_api_handler.receive_command():
                case ControlCommands.PAUSE:
                    self.logger.info("Pausing...")
                    self.thread_controller.pause()
                case ControlCommands.RESUME:
                    self.logger.info("Resuming...")
                    self.thread_controller.resume()
                case ControlCommands.SHUTDOWN:
                    self.logger.info("Shutting down...")
                    self.thread_controller.shutdown()
