"""This file contains custom configuration resolvers for OmegaConf.

See: https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html
"""
import torch
from omegaconf import OmegaConf


def register_custom_resolvers() -> None:
    OmegaConf.register_new_resolver("torch.device", torch.device)  # Usage: ${torch.device: cuda:0}
