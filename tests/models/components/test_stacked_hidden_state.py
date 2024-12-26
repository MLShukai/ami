import pytest
import torch

from ami.models.components.sconv import SConv

BATCH = 4
DEPTH = 8
DIM = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1


class TestStackedHiddenState:
    @pytest.fixture
    def sconv(self):
        sconv = SConv(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT)
        return sconv

    def test_batch_len(self, sconv):
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (BATCH, DEPTH, LEN, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden = sconv(x, hidden[:, :, -1, :])
        assert x.shape == x_shape
        assert hidden.shape == hidden_shape

    def test_no_batch_len(self, sconv):
        x_shape = (LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (DEPTH, LEN, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden = sconv(x, hidden[:, -1, :])
        assert x.shape == x_shape
        assert hidden.shape == hidden_shape

    def test_batch_no_len(self, sconv):
        x_shape = (BATCH, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (BATCH, DEPTH, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden = sconv(x, hidden)
        assert x.shape == x_shape
        assert hidden.shape == hidden_shape

    def test_no_batch_no_len(self, sconv):
        x_shape = (DIM,)
        x = torch.randn(*x_shape)
        hidden_shape = (DEPTH, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden = sconv(x, hidden)
        assert x.shape == x_shape
        assert hidden.shape == hidden_shape

    def test_many_batch_shape(self, sconv):
        x_shape = (1, 2, 3, BATCH, LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (1, 2, 3, BATCH, DEPTH, LEN, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden = sconv(x, hidden[:, :, :, :, -1])
        assert x.shape == x_shape
        assert hidden.shape == hidden_shape
