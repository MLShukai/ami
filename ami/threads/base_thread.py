from abc import ABC

from ..logger import get_thread_logger
from .thread_types import ThreadTypes


class BaseThread(ABC):
    """Base class for all thread objects.

    You must define the `THREAD_TYPE` attribute in the subclass's class field.
    Override the :meth:`worker` method for the thread's program.

    NOTE: Cannot create multiple threads of the same type due to competition in the value sharing namespace.
    """

    THREAD_TYPE: ThreadTypes

    def __init__(self) -> None:
        """Constructs the class and sets the logger."""

        if not hasattr(self, "THREAD_TYPE"):
            raise NotImplementedError("Thread class must define `THREAD_TYPE` attribute.")

        self.logger = get_thread_logger(self.THREAD_TYPE, self.__class__.__name__)

    def worker(self) -> None:
        """The program for this thread.

        please override this method.
        """
        raise NotImplementedError

    def run(self) -> None:
        """Executes the worker thread, logging any exceptions that occur.

        This method calls the `worker` method and logs any exceptions
        that arise during its execution. After logging, the exception is
        re-raised to ensure that exception handling can occur further up
        the call stack.
        """
        try:
            self.worker()
        except Exception:
            self.logger.exception("An exception occurred in the worker thread.")
            raise
