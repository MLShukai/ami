from ami.threads.base_thread import attach_shared_objects_pool_to_threads
from ami.threads.inference_thread import InferenceThread
from ami.threads.main_thread import MainThread
from ami.threads.training_thread import TrainingThread


def setup_threads() -> tuple[MainThread, InferenceThread, TrainingThread]:
    """Instantiates main, inference, training threads and attach shared object
    pool to them."""
    mt = MainThread()
    it = InferenceThread()
    tt = TrainingThread()
    attach_shared_objects_pool_to_threads(mt, it, tt)
    return mt, it, tt
