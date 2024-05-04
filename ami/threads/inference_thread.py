from pathlib import Path

from typing_extensions import override

from ..data.utils import DataCollectorsDict
from ..interactions.interaction import Interaction
from ..models.utils import InferenceWrappersDict
from .background_thread import BackgroundThread
from .shared_object_names import SharedObjectNames
from .thread_types import ThreadTypes


class InferenceThread(BackgroundThread):

    THREAD_TYPE = ThreadTypes.INFERENCE

    def __init__(self, interaction: Interaction, data_collectors: DataCollectorsDict) -> None:
        """Constructs the inference thread class."""
        super().__init__()

        self.interaction = interaction
        self.data_collectors = data_collectors

        self.share_object(SharedObjectNames.DATA_USERS, data_collectors.get_data_users())

    def on_shared_objects_pool_attached(self) -> None:
        super().on_shared_objects_pool_attached()

        self.inference_models: InferenceWrappersDict = self.get_shared_object(
            ThreadTypes.TRAINING, SharedObjectNames.INFERENCE_MODELS
        )

        # Attaches the objects to agent.
        self.interaction.agent.attach_data_collectors(self.data_collectors)
        self.interaction.agent.attach_inference_models(self.inference_models)

    def worker(self) -> None:
        self.logger.info("Start inference thread.")

        self.interaction.setup()

        self.logger.debug("Start the interaction loop.")
        while self.thread_command_handler.manage_loop():
            self.interaction.step()

        self.logger.debug("End the interaction loop.")

        self.interaction.teardown()

        self.logger.info("End the trainig thread.")

    @override
    def save_state(self, path: Path) -> None:
        path.mkdir()
        self.interaction.save_state(path / "interaction")

    @override
    def load_state(self, path: Path) -> None:
        self.interaction.load_state(path / "interaction")

    @override
    def on_paused(self) -> None:
        self.interaction.on_paused()

    @override
    def on_resumed(self) -> None:
        self.interaction.on_resumed()
