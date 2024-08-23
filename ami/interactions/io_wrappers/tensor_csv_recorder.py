import csv
import time
from typing import Any

from torch import Tensor
from typing_extensions import override

from .base_io_wrapper import BaseIOWrapper


class TensorCSVRecorder(BaseIOWrapper[Tensor, Tensor]):
    """Recording 1d tensor to csv.

    This class provides functionality to record 1-dimensional tensors to a CSV file.
    Each element of the tensor corresponds to a column in the CSV file.

    You have to provide the headers corresponding to each tensor element.
    A timestamp column is automatically added to the beginning of each row.

    Args:
        filename (str): The name of the CSV file to write to.
        headers (list[str]): List of column headers for the tensor elements.
        timestamp_header (str, optional): Header for the timestamp column. Defaults to "timestamp".

    Note:
        The input tensor must be 1-dimensional and its size must match the number of provided headers.
    """

    @override
    def __init__(self, filename: str, headers: list[str], timestamp_header: str = "timestamp") -> None:
        super().__init__()
        self.filename = filename
        self.headers = [timestamp_header] + headers
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        with open(self.filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.headers)

    @override
    def wrap(self, input: Tensor) -> Tensor:
        assert input.ndim == 1
        assert input.numel() == len(self.headers) - 1

        self.record_input(input.tolist())
        return input

    def record_input(self, input_array: list[Any]) -> None:
        row = [time.time()] + input_array
        with open(self.filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
