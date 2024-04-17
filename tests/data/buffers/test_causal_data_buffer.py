import pytest
import torch
from torch.utils.data import TensorDataset

from ami.data.buffers.causal_data_buffer import CausalDataBuffer
from ami.data.step_data import DataKeys


class TestCausalDataBuffer:
    max_len = 16
    key_list = [DataKeys.OBSERVATION]
    step_data = {DataKeys.OBSERVATION: torch.randn(16)}

    def test__init__(self):
        mod = CausalDataBuffer.reconstructable_init(self.max_len, self.key_list)
        assert len(mod) == 0

    def test_add(self):
        mod = CausalDataBuffer.reconstructable_init(self.max_len, self.key_list)
        mod.add(self.step_data)
        mod.add(self.step_data)
        mod.add(self.step_data)
        assert len(mod) == 3

    def test_add_overflow(self):
        mod = CausalDataBuffer.reconstructable_init(self.max_len, self.key_list)
        for _ in range(32):
            mod.add(self.step_data)
        assert len(mod) == self.max_len

    def test_concatenate(self):
        mod1 = CausalDataBuffer.reconstructable_init(self.max_len, self.key_list)
        mod1.add(self.step_data)
        mod1.add(self.step_data)
        mod1.add(self.step_data)
        mod2 = CausalDataBuffer.reconstructable_init(self.max_len, self.key_list)
        mod2.add(self.step_data)
        mod2.add(self.step_data)
        mod2.add(self.step_data)
        mod2.add(self.step_data)
        mod2.add(self.step_data)
        mod1.concatenate(mod2)
        assert len(mod1) == 8

    def test_make_dataset(self):
        mod = CausalDataBuffer.reconstructable_init(self.max_len, self.key_list)
        mod.add(self.step_data)
        mod.add(self.step_data)
        mod.add(self.step_data)
        assert isinstance(mod.make_dataset(), TensorDataset)

    def test_save_and_load_state(self, tmp_path):
        mod = CausalDataBuffer.reconstructable_init(self.max_len, self.key_list)
        mod.add(self.step_data)

        data_dir = tmp_path / "data"
        mod.save_state(data_dir)
        assert (data_dir / "observation.pkl").exists()

        saved_buffer_dataset = mod.make_dataset()

        mod = mod.new()
        mod.load_state(data_dir)

        loaded_buffer_dataset = mod.make_dataset()

        assert len(loaded_buffer_dataset) == len(saved_buffer_dataset) == 1
        assert torch.equal(loaded_buffer_dataset[:][0], saved_buffer_dataset[:][0])
