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
        assert torch.allclose(out.stddev, torch.nn.functional.softplus(torch.ones_like(out.stddev)))

    def test_squeeze_feature_dim(self):
        with pytest.raises(AssertionError):
            # out_features must be 1.
            FullyConnectedNormal(10, 2, squeeze_feature_dim=True)

        # `squeeze_feature_dim` default false.
        FullyConnectedNormal(10, 2)

        net = FullyConnectedNormal(10, 1, squeeze_feature_dim=True)
        x = torch.randn(10)
        out = net(x)
        assert out.sample().shape == ()
