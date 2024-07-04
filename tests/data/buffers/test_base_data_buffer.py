import pytest
import torch
from torch.utils.data import TensorDataset
from typing_extensions import Self

from ami.data.buffers.base_data_buffer import BaseDataBuffer
from ami.data.step_data import DataKeys, StepData
from tests.helpers import DataBufferImpl


class TestDataBuffer:
    @pytest.fixture
    def data_buffer(self) -> DataBufferImpl:
        return DataBufferImpl.reconstructable_init()

    @pytest.fixture
    def data_buffer_added(self, data_buffer: DataBufferImpl) -> DataBufferImpl:
        step_data = StepData({DataKeys.OBSERVATION: torch.randn(10)})
        data_buffer.add(step_data)
        return data_buffer

    def test_new(self, data_buffer_added: DataBufferImpl) -> None:

        new = data_buffer_added.new()
        assert new is not data_buffer_added
        assert new.obs == []
        assert new.obs != data_buffer_added.obs
