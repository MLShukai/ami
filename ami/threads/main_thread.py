from .base_thread import BaseThread
from .shared_object_names import SharedObjectNames
from .thread_control import ThreadController
from .thread_types import ThreadTypes


class MainThread(BaseThread):
    """Implements the main thread functionality within the ami system.

    Shares the thread command handlers with the training and inference
    threads.
    """

    THREAD_TYPE = ThreadTypes.MAIN

    def __init__(self) -> None:
        super().__init__()

        self.thread_controller = ThreadController()
        self.share_object(SharedObjectNames.THREAD_COMMAND_HANDLERS, self.thread_controller.handlers)
