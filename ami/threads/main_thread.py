from .base_thread import BaseThread
from .shared_object_pool import SharedObjectNames
from .thread_control import ThreadCommandHandler, ThreadController
from .thread_types import ThreadTypes


class MainThread(BaseThread):

    THREAD_TYPE = ThreadTypes.MAIN

    def __init__(self) -> None:
        super().__init__()

        self.thread_controller = ThreadController()

    def on_shared_object_pool_attached(self) -> None:
        super().on_shared_object_pool_attached()

        self.share_object(
            SharedObjectNames.THREAD_COMMAND_HANDLERS,
            {
                ThreadTypes.TRAINING: ThreadCommandHandler(self.thread_controller),
                ThreadTypes.INFERENCE: ThreadCommandHandler(self.thread_controller),
            },
        )
