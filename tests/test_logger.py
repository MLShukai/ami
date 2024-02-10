import logging

import pytest

from ami.logger import (
    ThreadTypes,
    get_inference_thread_logger,
    get_main_thread_logger,
    get_thread_logger,
    get_training_thread_logger,
)


def test_get_thread_logger():
    for e in ThreadTypes:
        get_thread_logger(e, "a")


@pytest.mark.parametrize("name", ["", ".test", "test.", "try..test"])
def test_invalid_logger_name(name: str):

    with pytest.raises(ValueError):
        get_thread_logger(ThreadTypes.MAIN, name)


def test_specified_thread_logger():
    main = get_main_thread_logger("a")
    assert main.name == "main.a"

    inference = get_inference_thread_logger("a")
    assert inference.name == "inference.a"

    training = get_training_thread_logger("a")
    assert training.name == "training.a"
