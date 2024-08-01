import random

import pytest
import torch

from ami.models.components.drop_path import DropPath


class TestDropPath:
    @pytest.mark.parametrize("input_n_dims", [2, 3, 4])
    @pytest.mark.parametrize("input_max_n_channels", [64])
    @pytest.mark.parametrize("drop_prob", [-1e-5, 0.0, 0.25, 0.5, 0.75, 1.0])
    @pytest.mark.parametrize("n_loops_for_convergence", [999])
    @pytest.mark.parametrize("allowable_difference", [0.2])
    def test_drop_path(
        self,
        input_n_dims: int,
        input_max_n_channels: int,
        drop_prob: float,
        n_loops_for_convergence: int,
        allowable_difference: float,
    ):
        if drop_prob < 0.0 or 1.0 <= drop_prob:
            with pytest.raises(AssertionError):
                drop_path = DropPath(drop_prob)
        else:
            drop_path = DropPath(drop_prob)

            input_size = torch.Size([random.randint(1, input_max_n_channels) for _ in range(input_n_dims)])
            input_tensor = torch.randn(size=input_size, dtype=torch.cfloat)
            output_tensor = drop_path(input_tensor)

            # check output size
            assert output_tensor.size() == input_size, f"Expected shape {input_size}, but got {output_tensor.shape}"

            # Whether the number of elements dropped is close to the drop_prob
            n_dropped = 0
            for _ in range(0, n_loops_for_convergence):
                output_tensor = drop_path(input_tensor)
                n_dropped += (output_tensor == 0.0).sum()
            assert n_dropped / (n_loops_for_convergence * output_tensor.numel()) == pytest.approx(
                drop_prob, abs=allowable_difference
            )
