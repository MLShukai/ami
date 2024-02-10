from ..logger import get_main_thread_logger
from .base_threads import BaseMainThread
from .thread_control import ThreadController
from .web_api_handler import WebApiHandler


class MainThread(BaseMainThread):
    """Main thread class."""

    def __init__(self) -> None:
        super().__init__()

        self._logger = get_main_thread_logger(self.__class__.__name__)
        self._controller = ThreadController()
        self._handler = WebApiHandler(self._controller)

    def worker(self) -> None:
        self._handler.run()

    def on_shared_object_pool_attached(self) -> None:
        pass
