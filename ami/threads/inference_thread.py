from .background_thread import BackgroundThread
from .thread_types import ThreadTypes


class InferenceThread(BackgroundThread):

    THREAD_TYPE = ThreadTypes.INFERENCE
