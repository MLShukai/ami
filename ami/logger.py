import logging
from typing import Any, Mapping, MutableSequence

from .threads.thread_types import ThreadTypes, get_thread_name_from_type

# Integration with the `warning` module.
# https://docs.python.org/3.11/library/logging.html#integration-with-the-warnings-module
logging.captureWarnings(True)


def get_thread_logger(thread_type: ThreadTypes, name: str) -> logging.Logger:
    """Obtains a logger for a specific thread type and module name.

    Args:
        thread_type: The type of thread for which the logger is being obtained.
        name: The name of the module or component for which the logger is intended.

    Raises:
        ValueError: If the module name is empty, starts/ends with a dot, or contains consecutive dots.

    Returns:
        logging.Logger: The logger for the specified thread type and name.
    """
    if name == "":
        raise ValueError("Module name is empty")

    # ドット始まり・終わり・途中のドットが2つ以上はエラー
    if name.startswith(".") or name.endswith(".") or ".." in name:
        raise ValueError(f'Invalid module name: "{name}"')

    thread_name = get_thread_name_from_type(thread_type)

    return logging.getLogger(f"{thread_name}.{name}")


def get_main_thread_logger(name: str) -> logging.Logger:
    return get_thread_logger(ThreadTypes.MAIN, name)


def get_training_thread_logger(name: str) -> logging.Logger:
    return get_thread_logger(ThreadTypes.TRAINING, name)


def get_inference_thread_logger(name: str) -> logging.Logger:
    return get_thread_logger(ThreadTypes.INFERENCE, name)


def _create_tree(data: Mapping[str, Any] | MutableSequence[Any], indent: str = "") -> str:
    result = ""
    if isinstance(data, MutableSequence):
        for value in data:
            if isinstance(value, (Mapping | MutableSequence)):
                result += f"{indent}- "
                result += _create_tree(value, indent + "  ")
            else:
                result += f"{indent}- {value}\n"
    elif isinstance(data, Mapping):
        for key, value in data.items():
            if isinstance(value, (Mapping | MutableSequence)):
                result += f"{indent}{key}:\n"
                result += _create_tree(value, indent + "  ")
            else:
                result += f"{indent}{key}: {value}\n"
    return result


def display_nested_config(data: Mapping[str, Any] | MutableSequence[Any]) -> str:
    return _create_tree(data)
