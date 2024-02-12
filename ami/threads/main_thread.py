from .base_thread import BaseThread
from .thread_types import ThreadTypes


class MainThread(BaseThread):
    """Implements the main thread functionality within the ami system."""

    THREAD_TYPE = ThreadTypes.MAIN
