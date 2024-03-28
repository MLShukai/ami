import torch
from omegaconf import OmegaConf

from ami.omegaconf_resolvers import register_custom_resolvers


def test_resolovers():
    register_custom_resolvers()

    assert OmegaConf.create({"device": "${torch.device: cuda:0}"}).device == torch.device("cuda:0")
    assert OmegaConf.create({"dtype": "${torch.dtype: complex64}"}).dtype == torch.complex64
