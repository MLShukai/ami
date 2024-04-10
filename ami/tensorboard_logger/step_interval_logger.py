from typing import Any

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from .tensorboard_logger import TensorBoardLogger


class StepIntervalLogger(TensorBoardLogger):
    def __init__(
        self,
        log_dir: str,
        log_every_n_steps: int = 1,
        **tensorboard_kwds: Any,
    ):
        super().__init__(log_dir, **tensorboard_kwds)
        self.log_every_n_steps = log_every_n_steps
        self.global_step = 0

    @property
    def log_available(self) -> bool:
        return self.global_step % self.log_every_n_steps == 0

    def update(self) -> None:
        self.global_step += 1

    def log(self, tag: str, scalar: Tensor | float | int) -> None:
        if self.log_available:
            self.tensorboard.add_scalar(tag, scalar, self.global_step)
