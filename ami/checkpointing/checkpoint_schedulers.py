import time
from abc import ABC, abstractmethod

from typing_extensions import override

from .checkpointing import Checkpointing


class BaseScheduler(ABC):
    """Schedules the checkpointing process with the specified condition.

    Expected usage:

        It is assumed that the checkpoint saving process is performed immediately after `is_available` returns True.

        ```py
        while True:
            if scheduler.is_available():
                save_checkpoint()
        ```
    """

    checkpointing: Checkpointing  # インスタンス生成後に任意でセットする。`save_checkpoint`を呼び出すための暫定的な参照位置。

    @abstractmethod
    def is_available(self) -> bool:
        """Returns whether or not it is time to save the checkpoint."""
        ...


class FixedTimeIntervalScheduler(BaseScheduler):
    """Saving the checkpoints with fixed time interval (seconds)."""

    def __init__(self, interval: float) -> None:
        super().__init__()
        self.interval = interval

        self._last_available_time = float("-inf")

    @override
    def is_available(self) -> bool:
        if available := time.monotonic() - self._last_available_time > self.interval:
            self._last_available_time = time.monotonic()
        return available
