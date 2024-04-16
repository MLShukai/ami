import logging

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
