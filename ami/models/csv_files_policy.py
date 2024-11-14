import torch
import torch.nn as nn
from torch import Tensor

from ami.interactions.io_wrappers.tensor_csv_recorder import TensorCSVReader

from .components.deterministic import Deterministic


class CSVFilesPolicy(nn.Module):
    """Generates actions from CSV files.

    This class processes CSV files sequentially, yielding action
    tensors. It supports multiple CSV files, custom column headers, and
    optional limitations on the number of rows to read from each file.
    """

    def __init__(
        self,
        file_paths: list[str],
        column_headers: list[str],
        dtype: torch.dtype,
        device: torch.device | None = None,
        file_max_rows: int | list[int | None] | None = None,
        observation_ndim: int = 1,
    ) -> None:
        """Initialize the CSVFilesPolicy.

        Args:
            file_paths: List of paths to CSV files containing action data.
            column_headers: List of column headers to read from the CSV files.
            dtype: Desired data type for the output tensors.
            device: Device to store the tensors on. If None, uses the default device.
            file_max_rows: Maximum number of rows to read from each file.
                           If an integer, applies to all files.
                           If a list, should match the length of file_paths.
                           If None, reads all rows from each file.
            observation_ndim: Number of dimensions in the observation tensor.

        Raises:
            ValueError: If the length of file_max_rows doesn't match the number of files
                        when provided as a list.
        """
        super().__init__()
        self.file_paths = file_paths
        self.observation_ndim = observation_ndim
        self.dtype = dtype
        self.device = device
        self.column_headers = column_headers

        # process max_rows
        if file_max_rows is None or isinstance(file_max_rows, int):
            file_max_rows = [file_max_rows] * len(file_paths)
        else:
            if len(file_paths) != len(file_max_rows):
                raise ValueError("Length of max_rows must match the number of files!")

        # Initialize TensorCSVReader objects for each folder.
        self.csv_readers = [
            TensorCSVReader(file, column_headers, float, max_row) for file, max_row in zip(file_paths, file_max_rows)
        ]

        self.current_reader_index = 0

    @property
    def max_actions(self) -> int:
        """Returns the total number of actions across all CSV files."""
        return sum(r.max_rows for r in self.csv_readers)

    def forward(self, observation: Tensor, forward_dynamics_hidden: Tensor) -> Deterministic:
        """Generate the next action from the CSV files.

        Args:
            observation: The current observation tensor.
            forward_dynamics_hidden: The hidden state of the forward dynamics model.

        Returns:
            A Deterministic distribution containing the next action tensor.

        Raises:
            StopIteration: When all actions from all CSV files have been processed.
        """
        batch_shape = observation.shape[: -self.observation_ndim]

        while self.current_reader_index < len(self.file_paths):
            current_reader = self.csv_readers[self.current_reader_index]
            if not current_reader.is_finished:
                action = current_reader.read()
                expanded_shape = batch_shape + action.shape
                return Deterministic(action.to(self.device, self.dtype).expand(expanded_shape))
            else:
                self.current_reader_index += 1

        raise StopIteration("All actions have been processed")
