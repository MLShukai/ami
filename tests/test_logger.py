import pytest

from ami.logger import get_main_thread_logger


@pytest.mark.parametrize("name", ["", ".test", "test.", "try..test"])
def test_invalid_logger_name(name: str) -> None:

    with pytest.raises(ValueError):
        get_main_thread_logger(name)
