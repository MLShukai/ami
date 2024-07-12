import torch

from ami.models.action_predictor import ContinousActionPredictor
from ami.models.components.fully_connected_fixed_std_normal import DeterministicNormal


def test_continous_action_predictor_forward():
    dim_embed = 16
    dim_action = 4
    dim_hidden = 32

    model = ContinousActionPredictor(dim_embed, dim_action, dim_hidden)
    embed_obs = torch.randn(10, dim_embed)
    next_embed_obs = torch.randn(10, dim_embed)

    output = model.forward(embed_obs, next_embed_obs)

    assert isinstance(output, DeterministicNormal)
    assert output.sample().shape == (10, dim_action)
