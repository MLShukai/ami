"""もし設計の拡張で静的なクラス数が増えた場合はここに記述する必要がある．"""
from enum import Enum, auto


class ThreadTypes(Enum):
    """Enumerates all types of threads that are implemented."""

    MAIN = auto()
    INFERENCE = auto()
    TRAINING = auto()
