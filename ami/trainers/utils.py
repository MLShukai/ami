from collections import UserList
from pathlib import Path

from typing_extensions import override

from ami.checkpointing import SaveAndLoadStateMixin
from ami.threads import PauseResumeEventMixin

from ..data.utils import DataUsersDict
from ..models.utils import ModelWrappersDict
from .base_trainer import BaseTrainer


class TrainersList(UserList[BaseTrainer], SaveAndLoadStateMixin, PauseResumeEventMixin):
    """A custom list class for aggregating trainers, designed for integration
    within the `hydra` configuration framework.

    This class facilitates the configuration and management of multiple trainer
    instances in a unified manner, enabling sequential or conditional execution within
    training workflows.

    Example usage in a Hydra configuration file (`config.yaml`):

        ```yaml
        trainers:
          _target_: <path.to>.TrainersList
          - _target_: <path.to.trainer1>
          - _target_: <path.to.trainer2>
        ```

    Usage in a `TrainingThread`:

        ```python
        class TrainingThread:
            trainers: TrainersList

            def worker(self):

                while True:
                    trainer = self.trainers.get_next_trainer()
                    trainer.run()
        ```
    """

    def __init__(self, *trainers: BaseTrainer) -> None:
        super().__init__(trainers)

        self._current_index = 0

    def get_next_trainer(self) -> BaseTrainer:
        """Retrieves the next trainer instance in a round-robin fashion for
        training.

        Raises:
            RuntimeError: If the TrainersList is empty, indicating there are no trainers to return.

        Returns:
            BaseTrainer: The next trainer instance in the list to be used for training.
        """
        if (length := len(self)) == 0:
            raise RuntimeError("TrainersList is empty!")

        trainer = self[self._current_index]
        self._current_index = (self._current_index + 1) % length
        return trainer

    def attach_model_wrappers_dict(self, model_wrappers_dict: ModelWrappersDict) -> None:
        """Attaches model wrappers to each trainer instance in the list.

        Args:
            model_wrappers_dict: A dictionary of model wrapper objects to be attached to trainers.
        """
        for trainer in self:
            trainer.attach_model_wrappers_dict(model_wrappers_dict)

    def attach_data_users_dict(self, data_users_dict: DataUsersDict) -> None:
        """Attaches data users to each trainer instance in the list.

        Args:
            data_users_dict: A dictionary of data user objects to be attached to trainers.
        """
        for trainer in self:
            trainer.attach_data_users_dict(data_users_dict)

    @override
    def save_state(self, path: Path) -> None:
        """Saves the internal state to the `path`."""
        path.mkdir()
        for i, trainer in enumerate(self):
            trainer_path = path / str(i)
            trainer.save_state(trainer_path)

    @override
    def load_state(self, path: Path) -> None:
        """Loads the internal state from the `path`"""
        for i, trainer in enumerate(self):
            trainer_path = path / str(i)
            trainer.load_state(trainer_path)

    @override
    def on_paused(self) -> None:
        for trainer in self:
            trainer.on_paused()

    @override
    def on_resumed(self) -> None:
        for trainer in self:
            trainer.on_resumed()
