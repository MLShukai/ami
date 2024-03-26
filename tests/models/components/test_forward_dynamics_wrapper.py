import pytest
import torch

from ami.models.components.forward_dynamics import ForwardDynamics
from ami.models.components.forward_dynamics_wrapper import ForwardDynamicsWrapper
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

    @pytest.fixture
    def forward_dynamics_wrapper(self, forward_dynamics):
        forward_dynamics_wrapper = ForwardDynamicsWrapper(forward_dynamics)
        return forward_dynamics_wrapper

    def test_forward_batch(self, forward_dynamics_wrapper):
        obs_shape = (BATCH, LEN, DIM_OBS)
        obs = torch.randn(*obs_shape)
        action_shape = (BATCH, LEN, DIM_ACTION)
        action = torch.randn(*action_shape)
        hidden_shape = (BATCH, DEPTH, LEN, DIM)
        hidden = torch.randn(*hidden_shape)

        obs_hat_dist, hidden = forward_dynamics_wrapper(obs, action, hidden[:, :, -1, :])
        assert obs_hat_dist.sample().shape == obs_shape
        assert hidden.shape == hidden_shape

        obs_hat_dist, hidden = forward_dynamics_wrapper(obs, action, hidden[:, :, -1, :])
        assert obs_hat_dist.sample().shape == obs_shape
        assert hidden.shape == hidden_shape

    def test_forward_no_batch(self, forward_dynamics_wrapper):
        obs_shape = (LEN, DIM_OBS)
        obs = torch.randn(*obs_shape)
        action_shape = (LEN, DIM_ACTION)
        action = torch.randn(*action_shape)
        hidden_shape = (DEPTH, LEN, DIM)
        hidden = torch.randn(*hidden_shape)

        obs_hat_dist, hidden = forward_dynamics_wrapper(obs, action, hidden[:, -1, :])
        assert obs_hat_dist.sample().shape == obs_shape
        assert hidden.shape == hidden_shape

        obs_hat_dist, hidden = forward_dynamics_wrapper(obs, action, hidden[:, -1, :])
        assert obs_hat_dist.sample().shape == obs_shape
        assert hidden.shape == hidden_shape

    def test_infer_batch(self, forward_dynamics_wrapper):
        obs_shape = (BATCH, DIM_OBS)
        obs = torch.randn(*obs_shape)
        action_shape = (BATCH, DIM_ACTION)
        action = torch.randn(*action_shape)
        hidden_shape = (BATCH, DEPTH, DIM)
        hidden = torch.randn(*hidden_shape)

        inference = forward_dynamics_wrapper.create_inference()

        obs_hat_dist, hidden = inference(obs, action, hidden)
        assert obs_hat_dist.sample().shape == obs_shape
        assert hidden.shape == hidden_shape

        obs_hat_dist, hidden = inference(obs, action, hidden)
        assert obs_hat_dist.sample().shape == obs_shape
        assert hidden.shape == hidden_shape

    def test_infer_no_batch(self, forward_dynamics_wrapper):
        obs_shape = (DIM_OBS,)
        obs = torch.randn(*obs_shape)
        action_shape = (DIM_ACTION,)
        action = torch.randn(*action_shape)
        hidden_shape = (DEPTH, DIM)
        hidden = torch.randn(*hidden_shape)

        inference = forward_dynamics_wrapper.create_inference()

        obs_hat_dist, hidden = inference(obs, action, hidden)
        assert obs_hat_dist.sample().shape == obs_shape
        assert hidden.shape == hidden_shape

        obs_hat_dist, hidden = inference(obs, action, hidden)
        assert obs_hat_dist.sample().shape == obs_shape
        assert hidden.shape == hidden_shape
