import torch

from ami.models.components.fully_connected_value_head import FullyConnectedValueHead


class TestFullyConnectedValueHead:
    def test_forward(self):
        m = FullyConnectedValueHead(16)

        assert m(torch.randn(8, 16)).shape == (8, 1)

        # test squeeze value dim
        m = FullyConnectedValueHead(16, True)
        assert m(torch.randn(8, 16)).shape == (8,)
