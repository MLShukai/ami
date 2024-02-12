from ami.threads.inference_thread import InferenceThread
from ami.threads.main_thread import MainThread
from ami.threads.shared_object_pool import SharedObjectPool
from ami.threads.training_thread import TrainingThread


def setup_threads() -> tuple[SharedObjectPool, MainThread, InferenceThread, TrainingThread]:
    """Instantiates main, inference, training threads and attach shared object
    pool to them."""
    sop = SharedObjectPool()
    mt = MainThread()
    it = InferenceThread()
    tt = TrainingThread()
    mt.attach_shared_object_pool(sop)
    it.attach_shared_object_pool(sop)
    tt.attach_shared_object_pool(sop)
    return sop, mt, it, tt
