import pytest
import torch

from ami.tensorboard_loggers import (
    StepIntervalLogger,
    TensorBoardLogger,
    TimeIntervalLogger,
)


class TestTensorBoardLogger:
    @pytest.fixture
    def logger(self, tmp_path):
        return TensorBoardLogger(tmp_path)

    @pytest.fixture
    def test_log(self, logger: TensorBoardLogger):
        for _ in range(64):
            logger.log("test1", torch.randn(1).item())
            logger.log("test2", torch.randn(1))
            logger.update()

    def test_convert_hparams_dict_values_to_loggable(self, logger: TensorBoardLogger):
        class A:
            def __str__(self) -> str:
                return "hello"

        assert logger._convert_hparams_dict_values_to_loggable({"a": A(), "b": 0}) == {"a": "hello", "b": 0}

    def test_log_hyperparameter(self, logger: TensorBoardLogger):
        d = {"a": 1, "b": [2, 3, {"c": 4, "d": [5, 6]}], "e": torch.device("cpu")}
        logger.log_hyperparameters(d)

    def test_state_dict(self, logger: TensorBoardLogger):

        assert logger.state_dict() == {"global_step": 0}
        logger.update()
        assert logger.state_dict() == {"global_step": 1}

    def test_load_state_dict(self, logger: TensorBoardLogger):

        assert logger.state_dict() == {"global_step": 0}
        state = {"global_step": 3}
        logger.load_state_dict(state)
        assert logger.state_dict() == state


class TestTimeIntervalLogger:
    @pytest.fixture
    def logger(self, tmp_path):
        return TimeIntervalLogger(tmp_path, 0.0)

    def test_log(self, logger: TimeIntervalLogger):
        for _ in range(64):
            logger.log("test1", torch.randn(1).item())
            logger.log("test2", torch.randn(1))
            logger.update()


class TestStepIntervalLogger:
    @pytest.fixture
    def logger(self, tmp_path):
        return StepIntervalLogger(tmp_path, 1)

    def test_log(self, logger: StepIntervalLogger):
        for _ in range(64):
            logger.log("test1", torch.randn(1).item())
            logger.log("test2", torch.randn(1))
            logger.update()
