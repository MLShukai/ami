import time
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(
        self,
        log_dir: str,
        log_every_n_steps: int = 1,
        log_every_n_seconds: float = 0.0,
        **tensorboard_kwds: Any,
    ):
        self.tensorboard = SummaryWriter(log_dir=log_dir, **tensorboard_kwds)
        self.log_every_n_steps = log_every_n_steps
        self.log_every_n_seconds = log_every_n_seconds
        self.global_step = 0
        self.previous_log_time: float | None = None

    @property
    def log_available(self) -> bool:
        return (
            self.global_step % self.log_every_n_steps == 0
            or self.previous_log_time is None
            or time.perf_counter() - self.previous_log_time > self.log_every_n_seconds
        )

    def update(self) -> None:
        self.global_step += 1
        self.previous_log_time = time.perf_counter()

    def log(self, tag: str, scalar: Tensor | float | int) -> None:
        if self.log_available:
            self.tensorboard.add_scalar(tag, scalar, self.global_step)
            self.update()

    def log_hyperparameters(
        self, hparams: DictConfig | Mapping[str, Any], metrics: dict[str, Any] | None = None
    ) -> None:
        def union_dicts(ld: list[dict[str, Any]]) -> dict[str, Any]:
            return {k: v for d in ld for k, v in d.items()}

        def expand_dict(d: Mapping[str, Any]) -> Mapping[str, Any]:
            return union_dicts(
                [
                    {k: v} if not isinstance(v, dict) else {f"{k}.{k_}": v_ for k_, v_ in expand_dict(v).items()}
                    for k, v in d.items()
                ]
            )

        self.tensorboard.add_hparams(
            expand_dict(hparams),
            {"hp_params": -1} if metrics is None else expand_dict(metrics),
        )
