from .background_thread import BackgroundThread
from .thread_types import ThreadTypes


class TrainingnThread(BackgroundThread):

    THREAD_TYPE = ThreadTypes.TRAINING
