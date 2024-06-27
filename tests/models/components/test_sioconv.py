import pytest
import torch

from ami.models.components.sioconv import SioConv

BATCH = 4
DEPTH = 8
DIM = 16
NUM_HEAD = 4
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1
CHUNK_SIZE = 16


class TestSioConv:
    @pytest.fixture
    def sioconv(self):
        sioconv = SioConv(DEPTH, DIM, NUM_HEAD, DIM_FF_HIDDEN, DROPOUT, CHUNK_SIZE)
        return sioconv

    def test_sconv(self, sioconv):
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (BATCH, DEPTH, LEN, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden = sioconv(x, hidden[:, :, -1, :])
        assert x.shape == x_shape
        assert hidden.shape == hidden_shape

        x, hidden = sioconv(x, hidden[:, :, -1, :])
        assert x.shape == x_shape
        assert hidden.shape == hidden_shape
