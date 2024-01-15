"""This file contains all base data buffer class."""
import copy
from abc import ABC, abstractmethod
from typing import Any, Self

from torch.utils.data import Dataset

from ..step_data import StepData


class BaseDataBuffer(ABC):
    """Base class for all data buffer objects.

    Please use the `init` method as the constructor instead of `__init__`.
    To renew this class, `__init__` stores the constructor's `args` and `kwds`, which are then used in the `new` method.

    Implement the following methods:
    - `add`: To store a single step of data from the agent.
    - `concatenate`: To concatenate the current data with new data.
    - `make_dataset`: To create a PyTorch dataset class for training.
    """

    def __init__(self, *args: Any, **kwds: Any) -> None:
        """Stores constructor arguments for renewing the data buffer, and throw
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
    def make_dataset(self) -> Dataset[Any]:
        """Makes the dataset object for training.

        Returns:
            dataset: Dataset object for training.
        """
        raise NotImplementedError

    def new(self) -> Self:
        return self.__class__(*self._init_args, **self._init_kwds)
