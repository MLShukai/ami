import math
import time
from typing import Any

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from .tensorboard_logger import TensorBoardLogger


class TimeIntervalLogger(TensorBoardLogger):
    def __init__(
        self,
        log_dir: str,
        log_every_n_seconds: float = 0.0,
        **tensorboard_kwds: Any,
    ):
        super().__init__(log_dir, **tensorboard_kwds)
        self.log_every_n_seconds = log_every_n_seconds
        self.global_step = 0
        self.previous_log_time: float = -math.inf

    @property
    def log_available(self) -> bool:
        return time.perf_counter() - self.previous_log_time > self.log_every_n_seconds

    def update(self) -> None:
        self.global_step += 1
        self.previous_log_time = time.perf_counter()

    def log(self, tag: str, scalar: Tensor | float | int) -> None:
        if self.log_available:
            self.tensorboard.add_scalar(tag, scalar, self.global_step)
