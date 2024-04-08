import pytest
import torch
from omegaconf import DictConfig, ListConfig

from ami.tensorboard_logger import TensorBoardLogger


class TestTensorBoardLogger:
    @pytest.fixture
    def tensorboard_logger(self, tmp_path):
        return TensorBoardLogger(tmp_path)

    def test_log(self, tensorboard_logger: TensorBoardLogger):
        for _ in range(64):
            tensorboard_logger.log("test1", torch.randn(1).item())
            tensorboard_logger.log("test2", torch.randn(1))
            tensorboard_logger.update()

    def test_log_hyperparameter(self, tensorboard_logger: TensorBoardLogger):
        d = DictConfig({"a": 1, "b": 2})
        tensorboard_logger.log_hyperparameters(d)
        d = ListConfig([1, 2, 3])
        tensorboard_logger.log_hyperparameters(d)
