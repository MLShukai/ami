import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableSequence
from typing import Any

from omegaconf import DictConfig, ListConfig
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger(ABC):
    def __init__(self, log_dir: str, **tensorboard_kwds: Any):
        self.tensorboard = SummaryWriter(log_dir=log_dir, **tensorboard_kwds)

    @property
    @abstractmethod
    def log_available(self) -> bool:
        """Determines whether a data can be logged.

        Returns:
            bool: whether a data can be logged.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> None:
        """Updates current step."""
        raise NotImplementedError

    @abstractmethod
    def log(self, tag: str, scalar: Tensor | float | int) -> None:
        """logs a single scalar.

        Args:
            tag: tag of data.
            scalar: data to log.
        """
        raise NotImplementedError

    def _union_dicts(self, ld: list[dict[str, Any]]) -> dict[str, Any]:
        return {k: v for d in ld for k, v in d.items()}

    def _expand_dict(self, d: Mapping[str, Any] | MutableSequence[Any]) -> dict[str, Any]:
        if isinstance(d, Mapping):
            dict_list = []
            for k, v in d.items():
                if not isinstance(v, Mapping | MutableSequence):
                    dict_list.append({k: v})
                else:
                    dict_list.append({f"{k}.{k_}": v_ for k_, v_ in self._expand_dict(v).items()})

            return self._union_dicts(dict_list)
        elif isinstance(d, MutableSequence):
            dict_list = []
            for i, v in enumerate(d):
                if not isinstance(v, Mapping | MutableSequence):
                    dict_list.append({f"{i}": v})
                else:
                    dict_list.append({f"{i}.{k_}": v_ for k_, v_ in self._expand_dict(v).items()})
            return self._union_dicts(dict_list)

    def log_hyperparameters(self, hparams: DictConfig | ListConfig, metrics: dict[str, Any] | None = None) -> None:
        self.tensorboard.add_hparams(
            self._expand_dict(hparams),
            {"hp_params": -1} if metrics is None else self._expand_dict(metrics),
        )


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
        self.logged = False

    @property
    def log_available(self) -> bool:
        return time.perf_counter() - self.previous_log_time > self.log_every_n_seconds

    def update(self) -> None:
        self.global_step += 1
        if self.logged:
            self.previous_log_time = time.perf_counter()
            self.logged = False

    def log(self, tag: str, scalar: Tensor | float | int) -> None:
        if self.log_available:
            self.tensorboard.add_scalar(tag, scalar, self.global_step)
            self.logged = True


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
