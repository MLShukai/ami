import pytest
import torch

from ami.models.components.sioconv_multi_lr import SioConvMultiLR

BATCH = 4
DEPTH = 8
DIM = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1
NUM_HEAD = 4
LR_SCALE = 100.0
WEIGHT_DECAY = 0.1


class TestSioConvMultiLR:
    @pytest.fixture
    def sioconv_multi_lr(self):
        sioconv_multi_lr = SioConvMultiLR(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT, NUM_HEAD, LR_SCALE, WEIGHT_DECAY)
        return sioconv_multi_lr

    def test_sioconv_multi_lr(self, sioconv_multi_lr):
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (BATCH, DEPTH, LEN, DIM)
        hidden = torch.randn(*hidden_shape)

        x, hidden = sioconv_multi_lr(x, hidden[:, :, -1, :])
        assert x.shape == x_shape
        assert hidden.shape == hidden_shape

        x, hidden = sioconv_multi_lr(x, hidden[:, :, -1, :])
        assert x.shape == x_shape
        assert hidden.shape == hidden_shape

    def test_sioconv_multi_lr_backward(self, sioconv_multi_lr):
        x_shape = (BATCH, LEN, DIM)
        x = torch.randn(*x_shape)
        hidden_shape = (BATCH, DEPTH, LEN, DIM)
        hidden = torch.randn(BATCH, DEPTH, DIM)

        x, hidden = sioconv_multi_lr(x, hidden)
        x.sum().backward()
