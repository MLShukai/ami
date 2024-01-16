"""This file contains an abstract base class for all trainers."""
from abc import ABC, abstractmethod

from ..data.interfaces import DataUser
from ..data.utils import DataUsersDict
from ..models.base_model import BaseModel
from ..models.utils import InferencesDict, ModelsDict


class BaseTrainer(ABC):
    """Abstract base class for all trainers.

    Override the :meth:`train` method to implement the training process.

    DNN models and data buffers become available after the thread has started.
    """

    _models_dict: ModelsDict
    _inferences_dict: InferencesDict
    _data_users_dict: DataUsersDict

    def __init__(self) -> None:
        """Constructs the trainer class."""
        super().__init__()
        self._synchronized_model_names: set[str] = set()

    def attach_models_dict(self, models_dict: ModelsDict) -> None:
        """Attaches the models dict to this trainer."""
        self._models_dict = models_dict

    def attach_inferences_dict(self, inferences_dict: InferencesDict) -> None:
        """Attaches the inferences dict to this trainer."""
        self._inferences_dict = inferences_dict

    def attach_data_users_dict(self, data_users_dict: DataUsersDict) -> None:
        """Attaches the data users dictionary to this trainer."""
        self._data_users_dict = data_users_dict

    def _get_model(self, name: str) -> BaseModel:
        """Retrieves the model from the models dictionary by `name`."""
        if name not in self._models_dict:
            raise KeyError(f"The specified model name '{name}' does not exist.")
        return self._models_dict[name]

    def get_frozen_model(self, name: str) -> BaseModel:
        """Retrieves the parameter-frozen (with `requires_grad=False`) model
        used for inference in the training flow."""
        model = self._get_model(name)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        return model

    def get_training_model(self, name: str) -> BaseModel:
        """Retrieves the model to be trained by this trainer.

        If the specified model includes an inference model, it is
        automatically synchronized after training.
        """
        model = self._get_model(name)
        if name in self._inferences_dict:
            self._synchronized_model_names.add(name)
        for p in model.parameters():
            p.requires_grad = True
        model.train()

        return model

    def get_data_user(self, name: str) -> DataUser:
        """Retrieves the specified data user."""
        return self._data_users_dict[name]

    @abstractmethod
    def train(self) -> None:
        """Train the deep neural network models.

        Please build the models, optimizers, dataset, and other components in this method.
        This method is called repeatedly.

        Use :meth:`get_training_model` to retrieve the model that will be trained.
        Use :meth:`get_frozen_model` to obtain the model used exclusively for performing inference during training.
        Use :meth:`get_data_user` to obtain the data user class for dataset.

        After this method, :meth:`synchronize` to be called.
        """
        raise NotImplementedError

    def synchronize(self) -> None:
        """Synchronizes the trained models with their corresponding inference
        models."""
        for name in self._synchronized_model_names:
            self._sync_a_model(name)

    def _sync_a_model(self, name: str) -> None:
        """Synchronizes the trained state of a DNN model with its corresponding
        inference model.

        Args:
            name: The name of the model to be synchronized.
        """
        trained_model = self._models_dict[name]
        inference = self._inferences_dict[name]
        un_trained_model = inference.model

        # Set trained model to inference.
        trained_model.eval()
        inference.model = trained_model

        # Load state dict to un-trained model from trained.
        un_trained_model.load_state_dict(trained_model.state_dict())
        un_trained_model.train()

        # Re-set to models dict
        self._models_dict[name] = un_trained_model

    def run(self) -> None:
        """Runs the training process."""
        self.train()
        self.synchronize()
