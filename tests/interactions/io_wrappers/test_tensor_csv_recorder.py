import os

import pytest
import torch

from ami.interactions.io_wrappers.tensor_csv_recorder import (
    TensorCSVReader,
    TensorCSVRecorder,
)


class TestTensorCSVRecorder:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.csv_filename = str(tmp_path / "test_output.csv")
        self.headers = ["value1", "value2", "value3"]
        self.recorder = TensorCSVRecorder(self.csv_filename, self.headers)

    def test_wrap(self):
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = self.recorder.wrap(input_tensor)
        assert torch.equal(result, input_tensor)

    def test_csv_output(self):
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        self.recorder.wrap(input_tensor)

        assert os.path.exists(self.csv_filename)

        with open(self.csv_filename) as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert lines[0].strip() == "timestamp,value1,value2,value3"
        assert lines[1].strip().split(",")[1:] == ["1.0", "2.0", "3.0"]

    def test_invalid_input(self):
        with pytest.raises(AssertionError):
            self.recorder.wrap(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))  # 2D tensor

        with pytest.raises(AssertionError):
            self.recorder.wrap(torch.tensor([1.0, 2.0]))  # Wrong number of elements


class TestTensorCSVReader:
    @pytest.fixture
    def sample_csv(self, tmp_path):
        csv_path = tmp_path / "test_input.csv"
        with open(csv_path, "w") as f:
            f.write("timestamp,value1,value2,value3\n")
            f.write("1234567890,1.0,2.0,3.0\n")
            f.write("1234567891,4.0,5.0,6.0\n")
            f.write("1234567892,7.0,8.0,9.0\n")
        return str(csv_path)

    @pytest.fixture
    def csv_reader(self, sample_csv):
        return TensorCSVReader(
            file_path=sample_csv, column_headers=["value1", "value2", "value3"], value_converter=float, max_rows=None
        )

    def test_init(self, sample_csv):
        reader = TensorCSVReader(
            file_path=sample_csv, column_headers=["value1", "value2", "value3"], value_converter=float, max_rows=2
        )
        assert reader.file_path == sample_csv
        assert reader.max_rows == 2

    def test_current_row(self, csv_reader):
        assert csv_reader.current_row == 0
        csv_reader.read()
        assert csv_reader.current_row == 1

    def test_is_finished(self, csv_reader):
        assert not csv_reader.is_finished
        csv_reader.read()
        csv_reader.read()
        csv_reader.read()
        assert csv_reader.is_finished

    def test_read(self, csv_reader):
        tensor = csv_reader.read()
        assert torch.equal(tensor, torch.tensor([1.0, 2.0, 3.0]))

    def test_read_all_rows(self, csv_reader):
        tensors = []
        while not csv_reader.is_finished:
            tensors.append(csv_reader.read())

        assert len(tensors) == 3
        assert torch.equal(tensors[0], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(tensors[1], torch.tensor([4.0, 5.0, 6.0]))
        assert torch.equal(tensors[2], torch.tensor([7.0, 8.0, 9.0]))

    def test_read_after_finished(self, csv_reader):
        while not csv_reader.is_finished:
            csv_reader.read()

        with pytest.raises(StopIteration):
            csv_reader.read()

    def test_invalid_header(self, sample_csv):
        with pytest.raises(ValueError):
            TensorCSVReader(file_path=sample_csv, column_headers=["value1", "invalid_header"], value_converter=float)

    def test_max_rows(self, sample_csv):
        reader = TensorCSVReader(
            file_path=sample_csv, column_headers=["value1", "value2", "value3"], value_converter=float, max_rows=2
        )
        reader.read()
        reader.read()
        assert reader.is_finished

    def test_empty_csv(self, tmp_path):
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")

        with pytest.raises(ValueError, match="CSV file is empty."):
            TensorCSVReader(
                file_path=str(empty_csv), column_headers=["value1", "value2", "value3"], value_converter=float
            )
