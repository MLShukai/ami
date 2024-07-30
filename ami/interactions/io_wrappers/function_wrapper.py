from typing import Callable, Generic

from .base_io_wrapper import BaseIOWrapper, WrappedType, WrappingType


class FunctionIOWrapper(BaseIOWrapper[WrappingType, WrappedType], Generic[WrappingType, WrappedType]):
    """The wrapper that allows to use any function as an IO wrapper."""

    def __init__(self, wrap_function: Callable[[WrappingType], WrappedType]):
        self.wrap_function = wrap_function

    def wrap(self, input: WrappingType) -> WrappedType:
        return self.wrap_function(input)
