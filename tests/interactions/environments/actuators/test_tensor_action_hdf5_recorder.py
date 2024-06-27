from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import Tensor
from typing_extensions import override

from ami.interactions.environments.actuators.tensor_action_hdf5_recorder import (
    BaseActuator,
    TensorActionHDF5Recorder,
)


class DummyActuator(BaseActuator[Tensor]):
    """A dummy actuator for testing purposes."""

    def __init__(self) -> None:
        self.actions = []

    def operate(self, action: Tensor) -> None:
        self.actions.append(action)

    def reset(self) -> None:
        self.actions = []


class TestTensorActionHDF5Recorder:
    def test_recording(self, tmp_path: Path):
        """Test that TensorActionHDF5Recorder saves and loads actions
        correctly."""
        actuator = DummyActuator()
        file_path = tmp_path / "test_actions.h5"
        recorder = TensorActionHDF5Recorder(actuator, file_path)
        recorder.setup()
        # Create some dummy tensor actions
        actions = [torch.tensor([i], dtype=torch.float32) for i in range(10)]

        # Record actions
        for action in actions[:5]:
            recorder.operate(action)

        # Flush remaining actions to file
        recorder.on_paused()
        recorder.on_resumed()

        for action in actions[5:]:
            recorder.operate(action)

        recorder.teardown()

        # Read the recorded actions from the HDF5 file
        with h5py.File(recorder._file_path, "r") as f:
            datasets = list(f.keys())
            recorded_tensors = [torch.tensor(f[ds][:]) for ds in datasets]

        # Combine all recorded tensors into one tensor
        recorded_tensors = torch.cat(recorded_tensors, dim=0)

        # Check if the recorded tensors match the original actions
        expected_tensors = torch.stack(actions).numpy()
        np.testing.assert_equal(recorded_tensors, expected_tensors)
