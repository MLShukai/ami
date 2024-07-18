import pytest
import torch

from ami.models.action_predictor import ContinousActionPredictor
from ami.models.components.fully_connected_fixed_std_normal import DeterministicNormal


@pytest.mark.parametrize("dim_embed", [16])
@pytest.mark.parametrize("dim_action", [4])
@pytest.mark.parametrize("dim_hidden", [32])
def test_continous_action_predictor_forward(
    dim_embed: int,
    dim_action: int,
    dim_hidden: int,
):

    model = ContinousActionPredictor(dim_embed, dim_action, dim_hidden)
    embed_obs = torch.randn(10, dim_embed)
    next_embed_obs = torch.randn(10, dim_embed)

    output = model.forward(embed_obs, next_embed_obs)

    assert isinstance(output, DeterministicNormal)
    assert output.sample().shape == (10, dim_action)
