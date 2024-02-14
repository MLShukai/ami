from typing import Any

from ..data.utils import DataUsersDict
from ..models.utils import ModelWrappersDict
from .base_trainer import BaseTrainer


class TrainersList(list[BaseTrainer]):
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

    _current_index = 0

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
