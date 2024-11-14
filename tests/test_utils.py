import pytest
import torch
from torch.testing import assert_close

from ami.utils import min_max_normalize


def test_min_max_normalize():
    # Basic case
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    assert_close(min_max_normalize(x), expected)

    # Custom range
    assert_close(min_max_normalize(x, new_min=-1, new_max=1), torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]))

    # 2D tensor
    x_2d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected_2d = torch.tensor([[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]])
    assert_close(min_max_normalize(x_2d), expected_2d)

    # Normalize along specific dimension
    assert_close(min_max_normalize(x_2d, dim=0), torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    assert_close(min_max_normalize(x_2d, dim=1), torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]))
