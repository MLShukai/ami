import pytest
import torch
from tensordict import TensorDict

from ami.trainers.components.transpose_and_stack_collator import (
    transpose_and_stack_collator,
)


class TestTransposeAndStackCollator:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_tensor_batch(self, batch_size: int):
        """Test collation of tensor batches."""
        # Create sample batch with observation and hidden state
        batch_items = [(torch.randn(3, 32, 32), torch.randn(8, 64)) for _ in range(batch_size)]

        result = transpose_and_stack_collator(batch_items)

        assert isinstance(result, tuple)
        assert len(result) == 2
        # Check observation tensor
        assert result[0].shape == (batch_size, 3, 32, 32)
        # Check hidden state tensor
        assert result[1].shape == (batch_size, 8, 64)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_tensordict_batch(self, batch_size: int):
        """Test collation of tensordict batches."""
        # Create sample batch with multimodal observations
        batch_items = [
            (
                TensorDict(
                    {
                        "image": torch.randn(3, 32, 32),
                        "audio": torch.randn(16),
                    },
                    [],
                ),
                torch.randn(8, 64),  # hidden state
            )
            for _ in range(batch_size)
        ]

        result = transpose_and_stack_collator(batch_items)

        assert isinstance(result, tuple)
        assert len(result) == 2
        # Check observation tensordict
        assert isinstance(result[0], TensorDict)
        assert result[0]["image"].shape == (batch_size, 3, 32, 32)
        assert result[0]["audio"].shape == (batch_size, 16)
        # Check hidden state tensor
        assert result[1].shape == (batch_size, 8, 64)
