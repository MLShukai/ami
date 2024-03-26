import pytest
import torch

from ami.models.components.forward_dynamics import ForwardDynamics
from ami.models.components.sconv import SConv

BATCH = 4
DEPTH = 8
DIM = 16
DIM_OBS = 16
DIM_ACTION = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1


class TestSconv:
    @pytest.fixture
    def sconv(self):
        sconv = SConv(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT)
        return sconv

    @pytest.fixture
    def forward_dynamics(self, sconv):
        forward_dynamics = ForwardDynamics(sconv, DIM, DIM_OBS, DIM_ACTION)
        return forward_dynamics

    def test_forward_dynamics(self, forward_dynamics):
        obs_shape = (BATCH, LEN, DIM_OBS)
        obs = torch.randn(*obs_shape)
        action_shape = (BATCH, LEN, DIM_ACTION)
        action = torch.randn(*action_shape)
        hidden_shape = (BATCH, DEPTH, LEN, DIM)
        hidden = torch.randn(*hidden_shape)

        obs_hat_dist, hidden = forward_dynamics(obs, action, hidden[:, :, -1, :])
        assert obs_hat_dist.sample().shape == obs_shape
        assert hidden.shape == hidden_shape

        obs_hat_dist, hidden = forward_dynamics(obs, action, hidden[:, :, -1, :])
        assert obs_hat_dist.sample().shape == obs_shape
        assert hidden.shape == hidden_shape
