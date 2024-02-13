from __future__ import annotations

import threading

from .thread_types import ThreadTypes


class ThreadController:
    """The controller class for sending commands from the main thread to
    background threads."""

    def __init__(self) -> None:
        """Construct this class."""
        self._shutdown_event = threading.Event()
        self._resume_event = threading.Event()  # For pause and resume.

        # Training Thread とInference Thread 間でHandlerインスタンスを分離。
        self.training_handler = ThreadCommandHandler(self)
        self.inference_handler = ThreadCommandHandler(self)

        self.activate()
        self.resume()  # default resume.

    def shutdown(self) -> None:
        """Sets the shutdown and resume flags."""
        self.resume()
        self._shutdown_event.set()

    def activate(self) -> None:
        """Clears the shutdown flag."""
        self._shutdown_event.clear()

    def is_shutdown(self) -> bool:
        """Returns the shutdown flag."""
        return self._shutdown_event.is_set()

    def resume(self) -> None:
        """Sets the resume flag."""
        self._resume_event.set()

    def pause(self) -> None:
        """Clears the resume flag."""
        self._resume_event.clear()

    def is_resumed(self) -> bool:
        """Returns the resume flag."""
        return self._resume_event.is_set()

    def wait_for_resume(self, timeout: float = 1.0) -> bool:
        """Waits for the resume event or times out after `timeout` seconds.

        Args:
            timeout: The maximum time to wait for the event, in seconds.

        Returns:
            bool: True if the event was set (resumed) before the timeout, False otherwise (timed out).
        """
        return self._resume_event.wait(timeout)

    @property
    def handlers(self) -> dict[ThreadTypes, ThreadCommandHandler]:
        """Creates the thread command handlers dict for registering to shared
        object pool."""
        handlers = {
            ThreadTypes.TRAINING: self.training_handler,
            ThreadTypes.INFERENCE: self.inference_handler,
        }
        return handlers


class ThreadCommandHandler:
    """Handles commands for thread management, facilitating communication and
    control between the main thread and background threads."""

    def __init__(self, controller: ThreadController, check_resume_interval: float = 1.0) -> None:
        """Constructs the ThreadCommandHandler class.

        Args:
            controller: An instance of the ThreadController class used for managing thread states.
            check_resume_interval: The interval, in seconds, at which the resume event is checked in the `manage_loop`.
        """
        self._controller = controller
        self.check_resume_interval = check_resume_interval

    def is_active(self) -> bool:
        """Checks if the managed thread should continue running."""
        return not self._controller.is_shutdown()

    def stop_if_paused(self) -> None:
        """Pauses the execution of the current thread until a resume command is
        received or the thread is stopped.

        This method periodically checks for a resume signal at intervals
        defined by `check_resume_interval` and stops executing if the
        thread is no longer active.
        """
        while self.is_active() and not self._controller.wait_for_resume(self.check_resume_interval):
            pass

    def manage_loop(self) -> bool:
        """Manages the infinite loop, blocking during pause states and
        returning the thread's activity flag.

        This method facilitates the implementation of a pause-resume mechanism within a running loop.
        Use this in a while loop to manage thread execution based on pause and resume commands.

        Example:
            ```python
            while thread_command_handler.manage_loop():
                ... # your process
            ```

        Returns:
            bool: True if the thread should continue executing, False if the thread is shutting down.
        """
        self.stop_if_paused()
        return self.is_active()
