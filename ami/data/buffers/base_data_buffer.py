"""This file contains all base data buffer class."""
import copy
from abc import ABC, abstractmethod
from typing import Any, Self

from torch.utils.data import Dataset

from ..step_data import StepData


class BaseDataBuffer(ABC):
    """Base class for all data buffer objects.

    Please use the `reconstructable_init` method as the constructor instead of `__init__`.
    To renew this class, `reconstructable_init` stores the constructor's `args` and `kwds`, which are then used in the `new` method.

    Implement the following methods:
    - `add`: To store a single step of data from the agent.
    - `concatenate`: To concatenate the current data with new data.
    - `make_dataset`: To create a PyTorch dataset class for training.
    """

    _init_args: tuple[Any, ...]
    _init_kwds: dict[str, Any]

    @classmethod
    def reconstructable_init(cls, *args: Any, **kwds: Any) -> Self:
        """Stores constructor arguments for renewing the data buffer, and throw
        them to :meth:`__init__`."""
        instance = cls(*copy.deepcopy(args), **copy.deepcopy(kwds))
        instance._init_args = args
        instance._init_kwds = kwds
        return instance

    @property
    def is_reconstructable(self) -> bool:
        return hasattr(self, "_init_args") and hasattr(self, "_init_kwds")

    def new(self) -> Self:
        if self.is_reconstructable:
            return self.__class__.reconstructable_init(*self._init_args, **self._init_kwds)
        else:
            raise RuntimeError(
                "Can not create new instance! Did you forget to use `reconstructable_init`"
                "instead of `__init__` when creating a instance?"
            )

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
