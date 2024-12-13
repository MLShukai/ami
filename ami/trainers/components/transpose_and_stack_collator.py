from typing import TypeAlias

import torch
from tensordict import TensorDict

TensorOrTensorDictType: TypeAlias = torch.Tensor | TensorDict


def transpose_and_stack_collator(
    batch_items: list[tuple[TensorOrTensorDictType, ...]]
) -> tuple[TensorOrTensorDictType, ...]:
    """Transposes a list of tuples and stacks each resulting column into a
    single tensor.

    Example:
        >>> tensors = [(tensor([1]), tensor([2])), (tensor([3]), tensor([4]))]
        >>> transpose_and_stack(tensors)
        (tensor([1, 3]), tensor([2, 4]))

    Args:
        batch_items: List of tuples containing tensors or tensor dicts to be processed.
            All tuples must have the same length.

    Returns:
        Tuple of stacked tensors or tensor dicts, where each element corresponds
        to a column in the original data.
    """
    return tuple(torch.stack(item) for item in zip(*batch_items))
