import torch
import torch.nn as nn
from torch.distributions import Normal

from ami.models.inverse_dynamics import InverseDynamics


class DummyEncoder(nn.Module):
    def forward(self, x):
        return x * 2


class DummyActionPredictor(nn.Module):
    def forward(self, embed, next_embed):
        mean = embed + next_embed
        std = torch.ones_like(mean)  # set standard deviations to 1
        return Normal(mean, std)


class TestInverseDynamics:
    def test_inverse_dynamics(self):
        observation_encoder = DummyEncoder()
        action_predictor = DummyActionPredictor()
        model = InverseDynamics(observation_encoder, action_predictor)

        obs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        next_obs = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        action_hat_dist = model(obs, next_obs)

        expected_mean = (obs * 2) + (next_obs * 2)
        assert torch.equal(action_hat_dist.mean, expected_mean), "Test failed!"
        assert torch.equal(action_hat_dist.stddev, torch.ones_like(expected_mean)), "Test failed!"
