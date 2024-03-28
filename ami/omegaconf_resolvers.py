"""This file contains custom configuration resolvers for OmegaConf.

See: https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html
"""
import torch
from omegaconf import OmegaConf

_registered = False


def register_custom_resolvers() -> None:
    global _registered

    if not _registered:
        OmegaConf.register_new_resolver("torch.device", torch.device)  # Usage: ${torch.device: cuda:0}
        OmegaConf.register_new_resolver("python.eval", eval)  # Usage: ${python.eval: 'lambda x:x * 2'}
        _registered = True
