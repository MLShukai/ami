import os

import pytest
import torch

from ami.interactions.io_wrappers.tensor_csv_recorder import TensorCSVRecorder


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
