from .background_thread import BackgroundThread
from .thread_types import ThreadTypes


class TrainingThread(BackgroundThread):

    THREAD_TYPE = ThreadTypes.TRAINING
