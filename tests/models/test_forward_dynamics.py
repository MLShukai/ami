import pytest
import torch
import torch.nn as nn

from ami.models.components.fully_connected_fixed_std_normal import (
    FullyConnectedFixedStdNormal,
)
from ami.models.components.sconv import SConv
from ami.models.forward_dynamics import (
    ForwardDynamcisWithActionReward,
    ForwardDynamics,
    PrimitiveForwardDynamics,
)

BATCH = 4
DEPTH = 8
DIM = 16
DIM_FF_HIDDEN = 32
LEN = 64
DROPOUT = 0.1
DIM_OBS = 32
DIM_ACTION = 8


class TestForwardDynamics:
    @pytest.fixture
    def core_model(self):
        sconv = SConv(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT)
        return sconv

    @pytest.fixture
    def observation_flatten(self):
        return nn.Identity()

    @pytest.fixture
    def action_flatten(self):
        return nn.Identity()

    @pytest.fixture
    def obs_action_projection(self):
        return nn.Linear(DIM_OBS + DIM_ACTION, DIM)

    @pytest.fixture
    def obs_hat_dist_head(self):
        return FullyConnectedFixedStdNormal(DIM, DIM_OBS)

    @pytest.fixture
    def forward_dynamics(
        self,
        observation_flatten,
        action_flatten,
        obs_action_projection,
        core_model,
        obs_hat_dist_head,
    ):
        return ForwardDynamics(
            observation_flatten, action_flatten, obs_action_projection, core_model, obs_hat_dist_head
        )

    def test_forward_dynamycs(self, forward_dynamics):
        obs_shape = (LEN, DIM_OBS)
        obs = torch.randn(*obs_shape)
        action_shape = (LEN, DIM_ACTION)
        action = torch.randn(*action_shape)
        hidden_shape = (DEPTH, LEN, DIM)
        hidden = torch.randn(*hidden_shape)

        obs_hat_dist, hidden = forward_dynamics(obs, hidden[:, -1, :], action)
        assert obs_hat_dist.sample().shape == obs_shape
        assert hidden.shape == hidden_shape


class TestForwardDynamicsWithActionReward:
    @pytest.fixture
    def core_model(self):
        sconv = SConv(DEPTH, DIM, DIM_FF_HIDDEN, DROPOUT)
        return sconv

    @pytest.fixture
    def observation_flatten(self):
        return nn.Identity()

    @pytest.fixture
    def action_flatten(self):
        return nn.Identity()

    @pytest.fixture
    def obs_action_projection(self):
        return nn.Linear(DIM_OBS + DIM_ACTION, DIM)

    @pytest.fixture
    def obs_hat_dist_head(self):
        return FullyConnectedFixedStdNormal(DIM, DIM_OBS)

    @pytest.fixture
    def action_hat_dist_head(self):
        return FullyConnectedFixedStdNormal(DIM, DIM_ACTION)

    @pytest.fixture
    def reward_head(self):
        return FullyConnectedFixedStdNormal(DIM, 1)

    @pytest.fixture
    def forward_dynamics(
        self,
        observation_flatten,
        action_flatten,
        obs_action_projection,
        core_model,
        obs_hat_dist_head,
        action_hat_dist_head,
        reward_head,
    ):
        return ForwardDynamcisWithActionReward(
            observation_flatten,
            action_flatten,
            obs_action_projection,
            core_model,
            obs_hat_dist_head,
            action_hat_dist_head,
            reward_head,
        )

    def test_forward_dynamycs(self, forward_dynamics):
        obs_shape = (LEN, DIM_OBS)
        obs = torch.randn(*obs_shape)
        action_shape = (LEN, DIM_ACTION)
        action = torch.randn(*action_shape)
        hidden_shape = (DEPTH, LEN, DIM)
        hidden = torch.randn(*hidden_shape)

        obs_hat_dist, action_hat_dist, reward_dist, hidden = forward_dynamics(obs, hidden[:, -1, :], action)
        assert obs_hat_dist.sample().shape == obs_shape
        assert action_hat_dist.sample().shape == action_shape
        assert reward_dist.sample().shape == (LEN, 1)
        assert hidden.shape == hidden_shape


class TestPrimitiveForwardDynamics:
    @pytest.fixture
    def core_model(self):
        return nn.Linear(DIM, DIM)

    @pytest.fixture
    def observation_flatten(self):
        return nn.Identity()

    @pytest.fixture
    def action_flatten(self):
        return nn.Identity()

    @pytest.fixture
    def obs_action_projection(self):
        return nn.Linear(DIM_OBS + DIM_ACTION, DIM)

    @pytest.fixture
    def obs_hat_dist_head(self):
        return FullyConnectedFixedStdNormal(DIM, DIM_OBS)

    @pytest.fixture
    def forward_dynamics(
        self,
        observation_flatten,
        action_flatten,
        obs_action_projection,
        core_model,
        obs_hat_dist_head,
    ):
        return PrimitiveForwardDynamics(
            observation_flatten,
            action_flatten,
            obs_action_projection,
            core_model,
            obs_hat_dist_head,
        )

    def test_forward_dynamycs(self, forward_dynamics):
        obs_shape = (LEN, DIM_OBS)
        obs = torch.randn(*obs_shape)
        action_shape = (LEN, DIM_ACTION)
        action = torch.randn(*action_shape)

        obs_hat_dist = forward_dynamics(obs, action)
        assert obs_hat_dist.sample().shape == obs_shape
