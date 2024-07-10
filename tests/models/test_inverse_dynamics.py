import torch
import torch.nn as nn

from ami.models.inverse_dynamics import InverseDynamics


class DummyEncoder(nn.Module):
    def forward(self, x):
        return x * 2


class DummyActionPredictor(nn.Module):
    def forward(self, embed, next_embed):
        return embed + next_embed


class TestInverseDynamics:
    def test_inverse_dynamics(self):
        observation_encoder = DummyEncoder()
        action_predictor = DummyActionPredictor()
        model = InverseDynamics(observation_encoder, action_predictor)

        obs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        next_obs = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        action_hat = model(obs, next_obs)

        expected_action_hat = (obs * 2) + (next_obs * 2)
        assert torch.equal(action_hat, expected_action_hat), "Test failed!"
