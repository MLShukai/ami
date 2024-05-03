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
        max_save_checkpoint_retry_attempts: int = 3,
    ) -> None:
        super().__init__()

        self.checkpoint_scheduler = checkpoint_scheduler
        self._host = address[0]
        self._port = address[1]
        self.thread_controller = ThreadController()
        self.web_api_handler = WebApiHandler(ThreadControllerStatus(self.thread_controller), self._host, self._port)
        self._timeout_for_all_threads_pause = timeout_for_all_threads_pause
        self.max_save_checkpoint_retry_attempts = max_save_checkpoint_retry_attempts

        self.share_object(SharedObjectNames.THREAD_COMMAND_HANDLERS, self.thread_controller.handlers)

    def worker(self) -> None:
        self.logger.info("Start main thread.")
        self.thread_controller.activate()

        self.logger.info(f"Serving system command at 'http://{self._host}:{self._port}'")
        self.web_api_handler.run_in_background()

        try:
            while not self.process_received_commands():

                if self.checkpoint_scheduler.is_available():
                    for i in range(self.max_save_checkpoint_retry_attempts + 1):
                        if self.save_checkpoint():
                            self.logger.info("Success to save checkpoint.")
                            break
                        else:
                            self.logger.warning(
                                f"Failed to save checkpoint. Attempts retry for {i+1} / {self.max_save_checkpoint_retry_attempts} times."
                            )

                time.sleep(0.001)

        except KeyboardInterrupt:
            self.logger.info("Shutting down by KeyboardInterrupt.")

        finally:
            self.logger.info("Shutting down...")
            self.thread_controller.shutdown()

        self.logger.info("End main thread.")

    def process_received_commands(self) -> bool:
        """Processes the received commands from web api handler.

        Returns `True` if the shutdown command is received.
        """
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

        return self.thread_controller.is_shutdown()

    def save_checkpoint(self) -> bool:
        """Saves a checkpoint after pausing the all background thread.

        Returns:
            bool: Whether or not to save checkpoint is failed by timeout.
        """

        self.logger.info("Saving checkpoint...")
        self.thread_controller.pause()

        if self.thread_controller.wait_for_all_threads_pause(self._timeout_for_all_threads_pause):
            self.logger.info("Success to pause the all background threads.")
        else:
            self.logger.error(
                f"Failed to pause the background threads in timeout {self._timeout_for_all_threads_pause} seconds."
            )
            self.thread_controller.resume()
            return False

        ckpt_path = self.checkpoint_scheduler.checkpointing.save_checkpoint()
        self.logger.info(f"Saved a checkpoint to '{ckpt_path}'")

        self.thread_controller.resume()
        return True
