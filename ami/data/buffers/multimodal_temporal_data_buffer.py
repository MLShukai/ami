import pickle
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias

import torch
from tensordict.tensordict import TensorDict
from torch import Tensor
from torch.utils.data import TensorDataset
from typing_extensions import Self, override

from ami.utils import Modality

from ..step_data import DataKeys, StepData
from .base_data_buffer import BaseDataBuffer

MultimodalObsType: TypeAlias = Mapping[Modality, Tensor]
MultimodalObsTensorDict: TypeAlias = TensorDict[Modality, Tensor]


@dataclass(frozen=True)
class TemporalDataStorage:
    observations: deque[MultimodalObsTensorDict]
    hiddens: deque[Tensor]
    added_times: deque[float]

    def to_dict(self) -> dict[str, deque[Any]]:
        return self.__dict__.copy()


class MultimodalTemporalDataBuffer(BaseDataBuffer):
    def __init__(
        self,
        max_len: int,
        observation_key: Literal[DataKeys.OBSERVATION, DataKeys.EMBED_OBSERVATION] = DataKeys.OBSERVATION,
    ):
        """Initializes data buffer.

        Args:
            max_len (int): max length of buffer.
            observation_key (Literal[DataKeys.OBSERVATION, DataKeys.EMBED_OBSERVATION], optional):
                Keyname of the observation. Defaults to DataKeys.EMBED_OBSERVATION.
        """
        super().__init__()

        self._max_len = max_len
        self._observation_key = observation_key

        temp_deque: deque[Any] = deque(maxlen=max_len)
        self._storage = TemporalDataStorage(
            observations=temp_deque.copy(),
            hiddens=temp_deque.copy(),
            added_times=temp_deque.copy(),
        )

    def __len__(self) -> int:
        """Returns current data length.

        Returns:
            int: current data length.
        """
        return len(self._storage.added_times)

    @override
    def add(self, step_data: StepData) -> None:
        """Add a single step of data.

        Args:
            step_data: A single step of data.
        """
        obs: MultimodalObsType = step_data[self._observation_key]
        obs = TensorDict({k: v for k, v in obs.items()}, batch_size=(), device="cpu")
        self._storage.observations.append(obs)
        self._storage.hiddens.append(Tensor(step_data[DataKeys.HIDDEN]).cpu())
        self._storage.added_times.append(time.time())

    @override
    def concatenate(self, new_data: Self) -> None:
        """Concatenates another buffer to this buffer.

        Args:
            new_data: A buffer to concatenate.
        """
        for key, value in new_data._storage.to_dict().items():
            self._storage.__dict__[key] += value

    @override
    def make_dataset(self) -> TensorDataset:
        """Make a TensorDataset from current buffer.

        Returns:
            TensorDataset: a TensorDataset created from current buffer.
        """
        return TensorDataset(torch.stack(list(self._storage.observations)), torch.stack(list(self._storage.hiddens)))

    def count_data_added_since(self, previous_get_time: float) -> int:
        """Counts the number of data points added since a specified time.

        Args:
            previous_get_time: The reference time to count from.

        Returns:
            int: The number of data points added since the specified time.
        """
        for i, t in enumerate(reversed(self._storage.added_times)):
            if t < previous_get_time:
                return i
        return len(self)

    @override
    def save_state(self, path: Path) -> None:
        super().save_state(path)
        path.mkdir()
        for key, value in self._storage.to_dict().items():
            with open(path / f"{key}.pkl", "wb") as f:
                pickle.dump(value, f)

    @override
    def load_state(self, path: Path) -> None:
        super().load_state(path)
        storage_dict = {}
        for key in self._storage.to_dict().keys():
            with open(path / f"{key}.pkl", "rb") as f:
                data = pickle.load(f)
            storage_dict[key] = deque(data, maxlen=self._max_len)
        self._storage = TemporalDataStorage(**storage_dict)
