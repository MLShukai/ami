"""This file contains all names (keys) of data, base data pool and its
implementations."""
import copy
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Self

from torch.utils.data import Dataset


class DataKeys(StrEnum):
    """Enumerates all the names of data obtained from the interaction between
    the environment and the agent."""

    PREVIOUS_ACTION = "previous_action"  # a_{t-1}
    OBSERVATION = "observation"  # o_t
    EMBED_OBSERVATION = "embed_observation"  # z_t
    ACTION = "action"  # a_t
    ACTION_LOG_PROBABILITY = "action_log_probability"
    VALUE = "value"  # v_t
    PREDICTED_NEXT_EMBED_OBSERVATION = "predicted_next_embed_observation"  # \hat{z}_{t+1}
    NEXT_OBSERVATION = "next_observation"  # o_{t+1}
    NEXT_EMBED_OBSERVATION = "next_embed_observation"  # z_{t+1}
    REWARD = "reward"  # r_{t+1}
    NEXT_VALUE = "next_value"  # v_{t+1}


class StepData(dict[str, Any]):
    """Dictionary that holds the data obtained from one step of the agent."""

    def copy(self) -> Self:
        """Return deep copied Self.

        Returns:
            self: deep copied data.
        """
        return copy.deepcopy(self)


class BaseDataPool(ABC):
    """Base class for all data pool objects.

    Please use the `init` method as the constructor instead of `__init__`. To renew this class, `__init__` stores the constructor's `args` and `kwds`, which are then used in the `new` method.

    Implement the following methods:
    - `add`: To store a single step of data from the agent.
    - `concatenate`: To concatenate the current data with new data.
    - `make_dataset`: To create a PyTorch dataset class for training.
    """

    def __init__(self, *args: Any, **kwds: Any) -> None:
        """Stores constructor arguments for renewing the data pool, and throw
        them to :meth:`init`."""
        self._init_args = copy.deepcopy(args)
        self._init_kwds = copy.deepcopy(kwds)
        self.init(*args, **kwds)

    def init(self, *args: Any, **kwds: Any) -> None:
        """User-defined constructor.

        The arguments must be constants as they are reused in the new
        instance.
        """
        pass

    @abstractmethod
    def add(self, step_data: StepData) -> None:
        """Stores a single step of data from the agent.

        Args:
            step_data: A single step of data from the agent.
        """
        raise NotImplementedError

    @abstractmethod
    def concatenate(self, new_data: Self) -> None:
        """Concatenates the current data with new data.

        Args:
            new_data: Data that is more recent than the current data.
        """
        raise NotImplementedError

    @abstractmethod
    def make_dataset(self) -> Dataset:
        """Makes the dataset object for training.

        Returns:
            dataset: Dataset object for training.
        """
        raise NotImplementedError

    def new(self) -> Self:
        return self.__class__(*self._init_args, **self._init_kwds)
