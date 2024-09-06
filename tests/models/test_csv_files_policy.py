import csv
import tempfile
from pathlib import Path

import pytest
import torch

from ami.models.csv_files_policy import CSVFilesPolicy


class TestCSVFilesPolicy:
    @pytest.fixture
    def sample_csv_files(self):
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        data = [[["a", "b", "c"], [1, 2, 3], [4, 5, 6]], [["a", "b", "c"], [7, 8, 9], [10, 11, 12]]]
        for i, file_data in enumerate(data):
            file_path = Path(temp_dir) / f"test_file_{i}.csv"
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(file_data)
            file_paths.append(file_path)
        return file_paths

    @pytest.fixture
    def csv_files_policy(self, sample_csv_files):
        return CSVFilesPolicy(
            file_paths=sample_csv_files, column_headers=["a", "b", "c"], dtype=torch.float32, device=torch.device("cpu")
        )

    def test_init(self, csv_files_policy, sample_csv_files):
        assert len(csv_files_policy.csv_readers) == len(sample_csv_files)
        assert csv_files_policy.current_reader_index == 0
        assert csv_files_policy.dtype == torch.float32
        assert csv_files_policy.device == torch.device("cpu")

    def test_max_actions(self, csv_files_policy):
        assert csv_files_policy.max_actions == 4  # 2 rows per file, 2 files

    def test_forward(self, csv_files_policy: CSVFilesPolicy):
        observation = torch.randn(3, 4)  # Dummy observation
        forward_dynamics_hidden = torch.randn(2, 2)  # Dummy hidden state

        # First action
        action1 = csv_files_policy.forward(observation, forward_dynamics_hidden)
        assert torch.allclose(action1.sample(), torch.tensor([1.0, 2.0, 3.0]))

        # Second action
        action2 = csv_files_policy.forward(observation, forward_dynamics_hidden)
        assert torch.allclose(action2.sample(), torch.tensor([4.0, 5.0, 6.0]))

        # Third action (from second file)
        action3 = csv_files_policy.forward(observation, forward_dynamics_hidden)
        assert torch.allclose(action3.sample(), torch.tensor([7.0, 8.0, 9.0]))

        # Fourth action
        action4 = csv_files_policy.forward(observation, forward_dynamics_hidden)
        assert torch.allclose(action4.sample(), torch.tensor([10.0, 11.0, 12.0]))

        # Should raise StopIteration after all actions are processed
        with pytest.raises(StopIteration):
            csv_files_policy.forward(observation, forward_dynamics_hidden)

    def test_file_max_rows(self, sample_csv_files):
        policy = CSVFilesPolicy(
            file_paths=sample_csv_files,
            column_headers=["a", "b", "c"],
            dtype=torch.float32,
            device=torch.device("cpu"),
            file_max_rows=[1, None],  # Read only 1 row from first file, all from second
        )
        assert policy.max_actions == 3  # 1 from first file, 2 from second

    def test_invalid_file_max_rows(self, sample_csv_files):
        with pytest.raises(ValueError):
            CSVFilesPolicy(
                file_paths=sample_csv_files,
                column_headers=["a", "b", "c"],
                dtype=torch.float32,
                device=torch.device("cpu"),
                file_max_rows=[1, 2, 3],  # More elements than files
            )

    def test_observation_ndim(self, sample_csv_files):
        policy = CSVFilesPolicy(
            file_paths=sample_csv_files,
            column_headers=["a", "b", "c"],
            dtype=torch.float32,
            device="cpu",
            observation_ndim=2,
        )
        observation = torch.randn(2, 4, 5)  # observatio shape (4, 5)
        forward_dynamics_hidden = torch.randn(2, 2)
        action = policy.forward(observation, forward_dynamics_hidden)
        assert action.sample().shape == (2, 3)  # Should match batch shape of observation
