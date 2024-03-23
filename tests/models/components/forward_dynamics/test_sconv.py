import pytest
import torch

from ami.models.components.forward_dynamics.sconv import SConv

BATCH = 4
DEPTH = 8
DIM = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1


class TestSconv:
    @pytest.fixture
    def sconv(self):
        sconv = SConv(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT)
        return sconv

    def test_sconv(self, sconv):
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (BATCH, DEPTH, LEN, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden = sconv(x, hidden[:, :, -1, :])
        assert x.shape == x_shape
        assert hidden.shape == hidden_shape

        x, hidden = sconv(x, hidden[:, :, -1, :])
        assert x.shape == x_shape
        assert hidden.shape == hidden_shape
