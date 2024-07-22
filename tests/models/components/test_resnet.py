import pytest
import torch

from ami.models.components.resnet import ResNetFF


class TestMultiEmbedding:
    @pytest.mark.parametrize(
        """
        batch,
        dim,
        dim_hidden,
        depth,
        """,
        [
            (3, 128, 256, 4),
            (6, 28, 56, 2),
        ],
    )
    def test_resnet_ff(self, batch, dim, dim_hidden, depth):
        mod = ResNetFF(dim, dim_hidden, depth)
        x = torch.randn(batch, dim)
        x = mod(x)
        assert x.shape == (batch, dim)
