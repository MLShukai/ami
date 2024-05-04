from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TypeAlias

from ..logger import get_main_thread_logger
from .thread_types import BACKGROUND_THREAD_TYPES, ThreadTypes

OnPausedCallbackType: TypeAlias = Callable[[], None]
OnResumedCallbackType: TypeAlias = Callable[[], None]


def dummy_on_paused() -> None:
    pass


def dummy_on_resumed() -> None:
    pass


class ThreadController:
    """The controller class for sending commands from the main thread to
    background threads.

    NOTE: **Only one thread can control this object.**
    """

    def __init__(self) -> None:
        """Construct this class."""
        self._shutdown_event = threading.Event()
        self._resume_event = threading.Event()  # For pause and resume.
        self._logger = get_main_thread_logger(self.__class__.__name__)

        # Thread間でHandlerインスタンスを分離。
        self.handlers: dict[ThreadTypes, ThreadCommandHandler] = dict()
        for thread_type in BACKGROUND_THREAD_TYPES:
            self.handlers[thread_type] = ThreadCommandHandler(self)

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

    def is_paused(self) -> bool:
        """Returns the flipped resume flag."""
        return not self.is_resumed()

    def wait_for_resume(self, timeout: float = 1.0) -> bool:
        """Waits for the resume event or times out after `timeout` seconds.

        Args:
            timeout: The maximum time to wait for the event, in seconds.

        Returns:
            bool: True if the event was set (resumed) before the timeout, False otherwise (timed out).
        """
        return self._resume_event.wait(timeout)

    def wait_for_shutdown(self, timeout: float = 1.0) -> bool:
        """Waits for the shutdown event or times out after `timeout`
        seconds."""
        return self._shutdown_event.wait(timeout)

    def wait_for_all_threads_pause(self, timeout: float | None = None) -> bool:
        """Waits for the all threads to pause.

        Args:
            timeout: Timeout seconds to wait for all threads to pause.

        Returns:
            bool: Whethers the all threads are paused or not.
        """

        tasks: dict[ThreadTypes, Future[bool]] = {}
        with ThreadPoolExecutor(max_workers=len(self.handlers)) as executor:
            for thread_type, hdlr in self.handlers.items():
                tasks[thread_type] = executor.submit(hdlr.wait_for_loop_pause, timeout)

        success = True
        for thread_type, tsk in tasks.items():
            if not (result := tsk.result()):
                self._logger.error(f"Timeout for waiting pause '{thread_type}' in {timeout} seconds.")
            success &= result
        return success


class ThreadCommandHandler:
    """Handles commands for thread management, facilitating communication and
    control between the main thread and background threads."""

    # 外部から定義されるコールバック関数
    on_paused: OnPausedCallbackType = dummy_on_paused
    on_resumed: OnResumedCallbackType = dummy_on_resumed

    def __init__(self, controller: ThreadController, check_resume_interval: float = 1.0) -> None:
        """Constructs the ThreadCommandHandler class.

        Args:
            controller: An instance of the ThreadController class used for managing thread states.
            check_resume_interval: The interval, in seconds, at which the resume event is checked in the `manage_loop`.
        """
        self._controller = controller
        self.check_resume_interval = check_resume_interval
        self._loop_pause_event = threading.Event()

    def is_active(self) -> bool:
        """Checks if the managed thread should continue running."""
        return not self._controller.is_shutdown()

    def stop_if_paused(self) -> None:
        """Pauses the execution of the current thread until a resume command is
        received or the thread is stopped.

        Executes the pause and resume event callbacks if system is paused.

        This method periodically checks for a resume signal at intervals
        defined by `check_resume_interval` and stops executing if the
        thread is no longer active.

        NOTE: 2024/05/04現在、スレッド処理の都合上、ごく稀に `controller.is_paused`が
            `False`を返した直後に`True`になり、一時停止イベントが呼び出されない場合がある。
            これは大きなバグの要因につながるため、将来的に直さなければならない。
            その可能性を最小限に抑えるために、pauseイベントの呼び出し処理をこのメソッドの外に書いてはならない。
        """
        self._loop_pause_event.clear()
        if self._controller.is_paused():  # Entering system state: `pause`
            self.on_paused()
            self._loop_pause_event.set()

        while not self._controller.wait_for_resume(self.check_resume_interval) and self.is_active():
            # `is_paused`と`wait_for_resume`の実行間隔を最小化。 whileの比較順序は変更しないこと
            pass

        if self._loop_pause_event.is_set():  # Exiting system state: `pause`, entering `resume`.
            self._loop_pause_event.clear()
            self.on_resumed()

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

        self.stop_if_paused()  # Blocking if system is `paused`

        return self.is_active()

    def is_loop_paused(self) -> bool:
        """Returns whether the background thread loop is paused or not."""
        return self._loop_pause_event.is_set()

    def wait_for_loop_pause(self, timeout: float | None = None) -> bool:
        """Waits for loop pause.

        Args:
            timeout: Time to wait for pause event.

        Returns:
            bool: Whethers the loop is paused or not.
        """
        return self._loop_pause_event.wait(timeout)


class ThreadControllerStatus:
    """Only reads the thread controller status."""

    def __init__(self, controller: ThreadController) -> None:

        self.is_shutdown = controller.is_shutdown
        self.is_paused = controller.is_paused
        self.is_resumed = controller.is_resumed
