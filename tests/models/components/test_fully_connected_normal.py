import pytest
import torch

from ami.models.components.fully_connected_normal import FullyConnectedNormal, Normal


class TestFullyConnectedNormal:
    @pytest.mark.parametrize(["dim_in", "dim_out", "batch_size"], [(8, 16, 1), (8, 1, 1)])
    def test_forward(self, dim_in, dim_out, batch_size):
        m = FullyConnectedNormal(dim_in, dim_out)
        data = torch.randn(batch_size, dim_in)
        out = m(data)
        assert isinstance(out, Normal)
        assert out.rsample().shape == (batch_size, dim_out)
