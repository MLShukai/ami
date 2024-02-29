from enum import StrEnum, auto


class SharedObjectNames(StrEnum):
    THREAD_COMMAND_HANDLERS = auto()
    DATA_USERS = auto()
    INFERENCE_MODELS = auto()
