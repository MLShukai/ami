import csv
import time
from typing import Any, Callable, Generic, TypeVar

import torch
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


ValueType = TypeVar("ValueType")


class TensorCSVReader(Generic[ValueType]):
    """Read CSV file and convert selected columns to a tensor.

    This class provides functionality to read data from a CSV file and convert
    selected columns into a PyTorch tensor. It allows for selective reading of
    columns and conversion of string values to a specified type.

    The class supports reading a specified number of rows and handles CSV files
    with a header row. It's designed to work with CSV files that have been created
    by TensorCSVRecorder or follow a similar format.

    Args:
        file_path (str): The path to the CSV file to read from.
        column_headers (list[str]): List of column headers to select from the CSV.
        value_converter (Callable[[str], ValueType]): A function to convert string values to the desired type.
        max_rows (int | None, optional): Maximum number of rows to read. If None, read all available rows. Defaults to None.

    Raises:
        ValueError: If the CSV file is empty, contains only a header, or if requested headers are not found.
    """

    def __init__(
        self,
        file_path: str,
        column_headers: list[str],
        value_converter: Callable[[str], ValueType],
        max_rows: int | None = None,
    ) -> None:
        super().__init__()

        self.file_path = file_path
        self.csv_file = open(file_path)
        self.csv_reader = csv.reader(self.csv_file)
        self.value_converter = value_converter

        total_rows = self._count_data_rows()

        if total_rows < 0:
            raise ValueError("CSV file is empty.")

        if max_rows is not None:
            if max_rows > total_rows:
                raise ValueError(f"Requested max_rows ({max_rows}) exceeds available data rows ({total_rows}).")
            self.max_rows = max_rows
        else:
            self.max_rows = total_rows

        file_headers = next(self.csv_reader)
        self.column_indices = self._get_column_indices(file_headers, column_headers)

    def _count_data_rows(self) -> int:
        with open(self.file_path) as f:
            return sum(1 for _ in f) - 1  # Subtract 1 to exclude header

    def _get_column_indices(self, file_headers: list[str], requested_headers: list[str]) -> list[int]:
        indices = []
        for h in requested_headers:
            if h in file_headers:
                indices.append(file_headers.index(h))
            else:
                raise ValueError(f"Requested header {h!r} not found in CSV.")
        return indices

    @property
    def current_row(self) -> int:
        return self.csv_reader.line_num - 1

    @property
    def is_finished(self) -> bool:
        return self.current_row >= self.max_rows

    def read(self) -> Tensor:
        """Read the next row from the CSV and return it as a tensor.

        Returns:
            Tensor: A tensor containing the converted values from the selected columns.

        Raises:
            StopIteration: When all rows have been read.
        """
        if self.is_finished:
            raise StopIteration("All rows have been read.")

        row_data = next(self.csv_reader)
        selected_data = [row_data[i] for i in self.column_indices]
        converted_data = [self.value_converter(d) for d in selected_data]
        return torch.tensor(converted_data)

    # def __getstate__(self) -> dict[str, Any]:
    #     """Prepare the object for pickling."""
    #     state = self.__dict__.copy()
    #     state["_reader_line_num"] = self.csv_reader.line_num
    #     del state["csv_file"]
    #     del state["csv_reader"]
    #     return state

    # def __setstate__(self, state: dict[str, Any]) -> None:
    #     """Restore the object from its pickled state."""
    #     reader_line_num = state.pop("_reader_line_num")
    #     self.__dict__.update(state)
    #     csv_file = open(self.file_path)
    #     csv_reader = csv.reader(csv_file)
    #     for _ in range(reader_line_num):
    #         next(csv_reader)
    #     self.csv_file = csv_file
    #     self.csv_reader = csv_reader

    def __del__(self) -> None:
        if hasattr(self, "csv_file"):
            self.csv_file.close()
