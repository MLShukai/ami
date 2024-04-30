import math
import time
from collections import ChainMap
from collections.abc import Mapping, MutableSequence
from typing import Any, TypeAlias

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import override

LoggableTypes: TypeAlias = Tensor | float | int | bool | str


class TensorBoardLogger:
    def __init__(self, log_dir: str, **tensorboard_kwds: Any):
        self.tensorboard = SummaryWriter(log_dir=log_dir, **tensorboard_kwds)
        self.global_step = 0

    @property
    def log_available(self) -> bool:
        """Determines whether a data can be logged.

        Returns:
            bool: whether a data can be logged.
        """
        return True

    def update(self) -> None:
        """Updates current step."""
        self.global_step += 1

    def log(self, tag: str, scalar: LoggableTypes) -> None:
        if self.log_available:
            self.tensorboard.add_scalar(tag, scalar, self.global_step)

    def _union_dicts(self, list_of_dict: list[dict[str, Any]]) -> dict[str, Any]:
        return dict(ChainMap(*list_of_dict))

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

    @staticmethod
    def _convert_hparams_dict_values_to_loggable(hparams: dict[str, Any]) -> dict[str, LoggableTypes]:
        """Converts the values of hparams dict to tensorboard loggable type.

        Currently, the unloggable type object is converted to `str`.
        """
        loggables: dict[str, LoggableTypes] = {}
        for key, value in hparams.items():
            if not isinstance(value, LoggableTypes):  # type: ignore
                value = str(value)
            loggables[key] = value
        return loggables

    def log_hyperparameters(self, hparams: Mapping[str, Any], metrics: dict[str, Any] | None = None) -> None:
        hparams = self._convert_hparams_dict_values_to_loggable(self._expand_dict(hparams))
        if metrics is None:
            metrics = {"hp_metrics": -1}
        else:
            metrics = self._convert_hparams_dict_values_to_loggable(self._expand_dict(metrics))

        self.tensorboard.add_hparams(hparams, metrics)

    def state_dict(self) -> dict[str, Any]:
        return {"global_step": self.global_step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.global_step = state_dict["global_step"]


class TimeIntervalLogger(TensorBoardLogger):
    def __init__(
        self,
        log_dir: str,
        log_every_n_seconds: float = 0.0,
        **tensorboard_kwds: Any,
    ):
        super().__init__(log_dir, **tensorboard_kwds)
        self.log_every_n_seconds = log_every_n_seconds
        self.previous_log_time: float = -math.inf
        self.logged = False

    @override
    @property
    def log_available(self) -> bool:
        return time.perf_counter() - self.previous_log_time > self.log_every_n_seconds

    @override
    def update(self) -> None:
        super().update()
        if self.logged:
            self.previous_log_time = time.perf_counter()
            self.logged = False

    @override
    def log(self, tag: str, scalar: LoggableTypes) -> None:
        super().log(tag, scalar)
        if self.log_available:
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

    @override
    @property
    def log_available(self) -> bool:
        return self.global_step % self.log_every_n_steps == 0
