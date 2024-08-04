import os

import pytest
import torch
from omegaconf import OmegaConf

from ami.omegaconf_resolvers import register_custom_resolvers, time_string_to_seconds


def test_resolovers():
    register_custom_resolvers()

    assert OmegaConf.create({"device": "${torch.device: cuda:0}"}).device == torch.device("cuda:0")
    assert OmegaConf.create({"eval": "${python.eval: 1 + 2 * 3 / 4}"}).eval == 2.5
    assert OmegaConf.create({"dtype": "${torch.dtype: complex64}"}).dtype == torch.complex64
    assert OmegaConf.create({"time": "${cvt_time_str: 10.0h}"}).time == 60 * 60 * 10
    assert OmegaConf.create({"cpu_count": "${os.cpu_count:}"}).cpu_count == os.cpu_count()


def test_time_string_to_seconds():
    # Valid cases
    assert time_string_to_seconds("1d") == 86400.0
    assert time_string_to_seconds("0.5h") == 1800.0
    assert time_string_to_seconds("30m") == 1800.0
    assert time_string_to_seconds("45s") == 45.0
    assert time_string_to_seconds("2w") == 1209600.0
    assert time_string_to_seconds("1mo") == 2628000.0
    assert time_string_to_seconds("1y") == 31536000.0
    assert time_string_to_seconds("100ms") == 0.1

    # Invalid format cases
    with pytest.raises(ValueError, match="Invalid time format: abc"):
        time_string_to_seconds("abc")

    with pytest.raises(ValueError, match="Invalid time format: 123"):
        time_string_to_seconds("123")

    with pytest.raises(ValueError, match="Invalid time format: h"):
        time_string_to_seconds("h")

    # Unknown time unit cases

    with pytest.raises(ValueError, match="Unknown time unit: nan"):
        time_string_to_seconds("1nan")
