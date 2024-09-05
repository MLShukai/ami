import logging

import pytest

from ami.logger import (
    ThreadTypes,
    display_nested_config,
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


def test_display_nested_config():
    # fmt: off
    data = {
        "key1": "value1",
        "key2": [
            "item1",
            "item2"
        ],
        "key3": {
            "nested_key": "nested_value"
        }
    }
    expected = (
        "key1: value1\n"
        "key2:\n"
        "  - item1\n"
        "  - item2\n"
        "key3:\n"
        "  nested_key: nested_value\n")
    # fmt: on
    assert display_nested_config(data) == expected
