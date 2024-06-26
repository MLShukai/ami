from ami.threads.main_thread import MainThread, ThreadTypes
from ami.threads.shared_object_names import SharedObjectNames


class TestMainThread:
    def test_object_sharing(self, checkpoint_scheduler):
        mt = MainThread(checkpoint_scheduler)

        # test can get.
        SharedObjectNames.THREAD_COMMAND_HANDLERS in mt.shared_objects_from_this_thread
