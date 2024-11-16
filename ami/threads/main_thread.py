import pickle
from pathlib import Path
from typing import TypeAlias

from typing_extensions import override

from ami import time

from ..checkpointing.checkpoint_schedulers import BaseCheckpointScheduler
from .base_thread import BaseThread
from .shared_object_names import SharedObjectNames
from .thread_control import ExceptionNotifier, ThreadController, ThreadControllerStatus
from .thread_types import (
    BACKGROUND_THREAD_TYPES,
    ThreadTypes,
    get_thread_name_from_type,
)
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
        max_uptime: float = float("inf"),
    ) -> None:
        """Constructs the main thread object.

        Args:
            checkpoint_scheduler: Scheduling checkpoint event.
            address: The tuple of host and port number for web api handler.
            timeout_for_all_threads_pause: Timeout seconds to wait for all threads pause. (for saving checkpoint.)
            max_attempts_to_pause_all_threads: Number of trials for failed attempts to pause all threads.
            max_uptime: Maximum system uptime. When this time is reached, the system will terminate.
        """
        super().__init__()

        self.checkpoint_scheduler = checkpoint_scheduler
        self._host = address[0]
        self._port = address[1]
        self.thread_controller = ThreadController()
        self.thread_controller.on_paused = self.on_paused
        self.thread_controller.on_resumed = self.on_resumed
        self.web_api_handler = WebApiHandler(ThreadControllerStatus(self.thread_controller), self._host, self._port)
        self._timeout_for_all_threads_pause = timeout_for_all_threads_pause
        self._max_attempts_to_pause_all_threads = max_attempts_to_pause_all_threads
        self._max_uptime = max_uptime

        self.share_object(SharedObjectNames.THREAD_COMMAND_HANDLERS, self.thread_controller.handlers)

    def on_shared_objects_pool_attached(self) -> None:
        self.exception_notifiers: dict[ThreadTypes, ExceptionNotifier] = {
            thread_type: self.get_shared_object(thread_type, SharedObjectNames.EXCEPTION_NOTIFIER)
            for thread_type in BACKGROUND_THREAD_TYPES
        }

    def worker(self) -> None:
        self.logger.info("Start main thread.")
        self.logger.info(f"Maxmum uptime is set to {self._max_uptime}.")
        self.thread_controller.activate()
        start_time = time.time()

        self.web_api_handler.run_in_background()

        try:
            while True:

                self.process_received_commands()

                if self.thread_controller.is_shutdown():
                    break

                if self.checkpoint_scheduler.is_available():
                    self.save_checkpoint()

                if self._max_uptime < (time.time() - start_time):
                    self.logger.info("Shutting down by reaching maximum uptime.")
                    break

                if self.check_background_threads_exception():
                    self.logger.error("An exception occurred. The system will terminate immediately.")
                    break

                time.fixed_sleep(0.001)

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
                case ControlCommands.SAVE_CHECKPOINT:
                    self.save_checkpoint()

    def save_checkpoint(self) -> None:
        """Saves a checkpoint after pausing the all background thread."""

        self.logger.info("Saving checkpoint...")

        for i in range(self._max_attempts_to_pause_all_threads):
            self.thread_controller.pause()

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

    def check_background_threads_exception(self) -> bool:
        """Checks the some exceptions has occurred in the background
        threads."""
        flag = False
        for thread_type, notififer in self.exception_notifiers.items():
            if notififer.is_raised():
                self.logger.error(f"The exception has occurred in the {get_thread_name_from_type(thread_type)} thread.")
                flag = True
        return flag

    @override
    def on_paused(self) -> None:
        super().on_paused()
        time.pause()

    @override
    def on_resumed(self) -> None:
        super().on_resumed()
        time.resume()

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        with open(path / "time_state.pkl", "wb") as f:
            pickle.dump(time.state_dict(), f)

    @override
    def load_state(self, path: Path) -> None:
        with open(path / "time_state.pkl", "rb") as f:
            time.load_state_dict(pickle.load(f))
