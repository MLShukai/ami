from enum import Enum, auto


class ThreadTypes(Enum):
    """Enumerates all types of threads that are implemented.

    NOTE: You need add new entry if static thread types are increased.
    """

    MAIN = auto()
    INFERENCE = auto()
    TRAINING = auto()
