import pytest
import torch

from ami.models.components.sioconvps import SioConvPS

BATCH = 4
DEPTH = 8
DIM = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1


class TestSioConvPS:
    @pytest.fixture
    def sioconvps(self):
        sioconvps = SioConvPS(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT)
        return sioconvps

    def test_sioconvps(self, sioconvps):
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (BATCH, DEPTH, LEN, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden = sioconvps(x, hidden[:, :, -1, :])
        assert x.shape == x_shape
        assert hidden.shape == hidden_shape

        x, hidden = sioconvps(x, hidden[:, :, -1, :])
        assert x.shape == x_shape
        assert hidden.shape == hidden_shape
