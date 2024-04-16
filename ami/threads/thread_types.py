from enum import Enum, auto


class ThreadTypes(Enum):
    """Enumerates all types of threads that are implemented.

    NOTE: You need add new entry if static thread types are increased.
    """

    MAIN = auto()
    INFERENCE = auto()
    TRAINING = auto()


BACKGROUND_THREAD_TYPES = {e for e in ThreadTypes if e is not ThreadTypes.MAIN}


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
