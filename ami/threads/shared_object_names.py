from enum import Enum, auto


class SharedObjectNames(str, Enum):
    THREAD_COMMAND_HANDLERS = auto()
    DATA_USERS = auto()
    INFERENCE_MODELS = auto()
