import logging

from .threads import ThreadTypes

# Integration with the `warning` module.
# https://docs.python.org/3.11/library/logging.html#integration-with-the-warnings-module
logging.captureWarnings(True)


def get_thread_name_from_type(thread_type: ThreadTypes) -> str:
    """Returns the thread name corresponding to the given `thread_type`.

    Args:
        thread_type: The type of thread as defined in ThreadTypes enum.

    Raises:
        NotImplementedError: If the name for the specified thread type is not implemented yet.
        ValueError: If an invalid thread type object is provided.

    Returns:
        str: The name of the thread corresponding to the given thread type.
    """
    match thread_type:
        case ThreadTypes.MAIN:
            return "main"
        case ThreadTypes.INFERENCE:
            return "inference"
        case ThreadTypes.TRAINING:
            return "training"
        case _:
            if thread_type in ThreadTypes:
                raise NotImplementedError(f"The name for specified thread type '{thread_type}' is not implemented yet.")
            raise ValueError(f"Invalid thread type object: {thread_type}")


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
