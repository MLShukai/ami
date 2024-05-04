import time
from typing import TypeAlias

from ..checkpointing.checkpoint_schedulers import BaseCheckpointScheduler
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

    def __init__(
        self,
        checkpoint_scheduler: BaseCheckpointScheduler,
        address: AddressType = ("0.0.0.0", 8391),
        timeout_for_all_threads_pause: float = 60.0,
        max_attempts_to_pause_all_threads: int = 3,
    ) -> None:
        """Constructs the main thread object.

        Args:
            checkpoint_scheduler: Scheduling checkpoint event.
            address: The tuple of host and port number for web api handler.
            timeout_for_all_threads_pause: Timeout seconds to wait for all threads pause. (for saving checkpoint.)
            max_attempts_to_pause_all_threads: Number of trials for failed attempts to pause all threads.
        """
        super().__init__()

        self.checkpoint_scheduler = checkpoint_scheduler
        self._host = address[0]
        self._port = address[1]
        self.thread_controller = ThreadController()
        self.web_api_handler = WebApiHandler(ThreadControllerStatus(self.thread_controller), self._host, self._port)
        self._timeout_for_all_threads_pause = timeout_for_all_threads_pause
        self._max_attempts_to_pause_all_threads = max_attempts_to_pause_all_threads

        self.share_object(SharedObjectNames.THREAD_COMMAND_HANDLERS, self.thread_controller.handlers)

    def worker(self) -> None:
        self.logger.info("Start main thread.")
        self.thread_controller.activate()

        self.logger.info(f"Serving system command at 'http://{self._host}:{self._port}'")
        self.web_api_handler.run_in_background()

        try:
            while True:

                self.process_received_commands()

                if self.thread_controller.is_shutdown():
                    break

                if self.checkpoint_scheduler.is_available():
                    self.save_checkpoint()

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

    def save_checkpoint(self) -> None:
        """Saves a checkpoint after pausing the all background thread."""

        self.logger.info("Saving checkpoint...")
        self.thread_controller.pause()

        for i in range(self._max_attempts_to_pause_all_threads):
            if self.thread_controller.wait_for_all_threads_pause(self._timeout_for_all_threads_pause):
                self.logger.info("Success to pause the all background threads.")
                break
            else:
                self.logger.warning(
                    f"Failed to pause the background threads in timeout {self._timeout_for_all_threads_pause} seconds."
                )
                self.logger.warning(f"Attempting retry {i+1} / {self._max_attempts_to_pause_all_threads} ...")
                self.thread_controller.resume()
        else:
            self.logger.error("Failed to save checkpoint because the thread pause process could not be completed... ")
            return

        ckpt_path = self.checkpoint_scheduler.checkpointing.save_checkpoint()
        self.logger.info(f"Saved a checkpoint to '{ckpt_path}'")

        self.thread_controller.resume()
