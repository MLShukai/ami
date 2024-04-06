from ..data.utils import DataUsersDict
from ..models.utils import ModelWrappersDict
from ..trainers.utils import TrainersList
from .background_thread import BackgroundThread
from .shared_object_names import SharedObjectNames
from .thread_types import ThreadTypes


class TrainingThread(BackgroundThread):

    THREAD_TYPE = ThreadTypes.TRAINING

    def __init__(self, trainers: TrainersList, models: ModelWrappersDict) -> None:
        """Constructs the training thread class with the trainers and
        models."""
        super().__init__()

        self.trainers = trainers
        self.models = models

        self.share_object(SharedObjectNames.INFERENCE_MODELS, models.inference_wrappers_dict)

    def on_shared_objects_pool_attached(self) -> None:
        super().on_shared_objects_pool_attached()

        self.data_users: DataUsersDict = self.get_shared_object(ThreadTypes.INFERENCE, SharedObjectNames.DATA_USERS)

        self.trainers.attach_data_users_dict(self.data_users)
        self.trainers.attach_model_wrappers_dict(self.models)

    def worker(self) -> None:
        self.logger.info("Starts the training thread.")

        while self.thread_command_handler.manage_loop():
            trainer = self.trainers.get_next_trainer()
            if trainer.is_trainable():
                self.logger.info(f"Running: {type(trainer).__name__} ...")
                trainer.run()

        self.logger.info("End the training thread.")
