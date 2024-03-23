import pytest
import torch
from torch.utils.data import TensorDataset

from ami.data.buffers.random_data_buffer import RandomDataBuffer
from ami.data.step_data import DataKeys


class TestRandomDataBuffer:
    max_len = 16
    key_list = [DataKeys.OBSERVATION]
    step_data = {DataKeys.OBSERVATION: torch.randn(16)}

    def test__init__(self):
        mod = RandomDataBuffer(self.max_len, self.key_list)
        assert len(mod) == 0

    def test_add(self):
        mod = RandomDataBuffer(self.max_len, self.key_list)
        mod.add(self.step_data)
        mod.add(self.step_data)
        mod.add(self.step_data)
        assert len(mod) == 3

    def test_add_overflow(self):
        mod = RandomDataBuffer(self.max_len, self.key_list)
        for _ in range(32):
            mod.add(self.step_data)
        assert len(mod) == self.max_len

    def test_concatenate(self):
        mod1 = RandomDataBuffer(self.max_len, self.key_list)
        mod1.add(self.step_data)
        mod1.add(self.step_data)
        mod1.add(self.step_data)
        mod2 = RandomDataBuffer(self.max_len, self.key_list)
        mod2.add(self.step_data)
        mod2.add(self.step_data)
        mod2.add(self.step_data)
        mod2.add(self.step_data)
        mod2.add(self.step_data)
        mod1.concatenate(mod2)
        assert len(mod1) == 8

    def test_make_dataset(self):
        mod = RandomDataBuffer(self.max_len, self.key_list)
        mod.add(self.step_data)
        mod.add(self.step_data)
        mod.add(self.step_data)
        assert isinstance(mod.make_dataset(), TensorDataset)
