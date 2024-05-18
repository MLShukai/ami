"""This file contains custom configuration resolvers for OmegaConf.

See: https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html
"""
import re

import torch
from omegaconf import OmegaConf

_registered = False


def register_custom_resolvers() -> None:
    global _registered

    if not _registered:
        OmegaConf.register_new_resolver("torch.device", torch.device)  # Usage: ${torch.device: cuda:0}
        OmegaConf.register_new_resolver("python.eval", eval)  # Usage: ${python.eval: 'lambda x:x * 2'}
        OmegaConf.register_new_resolver(
            "torch.dtype", convert_dtype_str_to_torch_dtype
        )  # Usage: ${torch.dtype: float32}
        OmegaConf.register_new_resolver("cvt_time_str", time_string_to_seconds)  # Usage: ${cvt_time_str:"1h"}

        _registered = True


def convert_dtype_str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert the string of dtype such as "float32" to `torch.dtype` object
    `torch.float32` .

    All dtypes are listed in https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype
    """
    if hasattr(torch, dtype_str):
        dtype = getattr(torch, dtype_str)
    else:
        raise ValueError(f"Dtype name {dtype_str} does not exists!")

    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Dtype name {dtype_str} is not dtype! Object: {dtype}")
    return dtype


TIME_UNITS = {
    "s": 1,  # seconds
    "m": 60,  # minutes
    "h": 3600,  # hours
    "d": 86400,  # days
    "w": 604800,  # weeks
    "mo": 2628000,  # months (30.44 days)
    "y": 31536000,  # years (365 days)
}


def time_string_to_seconds(time_str: str) -> float:
    """Converts the time string (e.g., 1.0h, 1d) to seconds."""

    # Extract the numeric and unit parts using regex
    matched = re.match(r"(\d*\.?\d+)([a-zA-Z]+)", time_str)
    if not matched:
        raise ValueError(f"Invalid time format: {time_str}")

    num_part = matched.group(1)
    unit_part = matched.group(2)

    num_value = float(num_part)

    if unit_part in TIME_UNITS:
        return num_value * TIME_UNITS[unit_part]
    else:
        raise ValueError(f"Unknown time unit: {unit_part}")
