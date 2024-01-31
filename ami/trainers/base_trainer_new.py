"""This file contains an abstract base class for all trainers."""
from abc import ABC, abstractmethod
from typing import TypeAlias

import torch.nn as nn

from ..data.interfaces import DataUser
from ..data.utils import DataUsersDict
from ..models.model_wrapper import ModelWrapper
from ..models.utils import InferenceWrappersDict, ModelWrappersDict

_model_wrapper_t: TypeAlias = ModelWrapper[nn.Module]


class BaseTrainer(ABC):
    """Abstract base class for all trainers.

    Override the following methods:
        - `on_model_wrappers_dict_attached`: To retrieve the models.
        - `on_data_users_dict_attached`: To retrieve the data users.
        - `train`: To implement the training process.

    DNN models and data buffers become available after the thread has started.
    """

    _model_wrappers_dict: ModelWrappersDict
    _inference_wrappers_dict: InferenceWrappersDict
    _data_users_dict: DataUsersDict

    def __init__(self) -> None:
        """Constructs the trainer class."""
        super().__init__()
        self._synchronized_model_names: set[str] = set()
        self._training_model_names: set[str] = set()
        self._frozen_model_names: set[str] = set()

    def attach_model_and_inference_wrappers_dict(
        self,
        model_wrappers_dict: ModelWrappersDict,
        inference_wrappers_dict: InferenceWrappersDict,
    ) -> None:
        """Attaches the model and inference wrappers dict to this trainer."""
        self._model_wrappers_dict = model_wrappers_dict
        self._inference_wrappers_dict = inference_wrappers_dict
        self.on_model_wrappers_dict_attached()

    def on_model_wrappers_dict_attached(self) -> None:
        """Callback method for when `model_wrappers_dict` is attached to the
        trainer.

        Override this method to retrieve DNN models.

        Use :meth:`get_training_model` to retrieve the model that will be trained.
        Use :meth:`get_frozen_model` to obtain the model used exclusively for performing inference during training.
        """
        pass

    def attach_data_users_dict(self, data_users_dict: DataUsersDict) -> None:
        """Attaches the data users dictionary to this trainer."""
        self._data_users_dict = data_users_dict
        self.on_data_users_dict_attached()

    def on_data_users_dict_attached(self) -> None:
        """Callback method when `data_users_dict` is attached to the trainer.

        Override this method to retrieve `DataUsers`.

        Use :meth:`get_data_user` to obtain the data user class for dataset.
        """
        pass

    def _get_model(self, name: str) -> _model_wrapper_t:
        """Retrieves the model from the models dictionary by `name`."""
        if name not in self._model_wrappers_dict:
            raise KeyError(f"The specified model name '{name}' does not exist.")
        return self._model_wrappers_dict[name]

    @staticmethod
    def make_model_frozen(model: _model_wrapper_t) -> _model_wrapper_t:
        """Makes the input model frozen (untrainable)."""
        model.freeze_model()
        model.eval()
        return model

    @staticmethod
    def make_model_trainable(model: _model_wrapper_t) -> _model_wrapper_t:
        """Makes the input model trainable."""
        model.unfreeze_model()
        model.train()
        return model

    def get_frozen_model(self, name: str) -> _model_wrapper_t:
        """Retrieves the parameter-frozen (with `requires_grad=False`) model
        used for inference in the training flow."""
        model = self._get_model(name)
        if name in self._training_model_names:
            raise RuntimeError(f"Model '{name}' is already used as training model!")
        self._frozen_model_names.add(name)
        return self.make_model_frozen(model)

    def get_training_model(self, name: str) -> _model_wrapper_t:
        """Retrieves the model to be trained by this trainer.

        If the specified model includes an inference model, it is
        automatically synchronized after training.
        """
        model = self._get_model(name)
        if name in self._frozen_model_names:
            raise RuntimeError(f"Model '{name}' is already used as frozen model!")
        if name in self._inference_wrappers_dict:
            self._synchronized_model_names.add(name)
        self._training_model_names.add(name)
        return self.make_model_trainable(model)

    def get_data_user(self, name: str) -> DataUser:
        """Retrieves the specified data user."""
        if name not in self._data_users_dict:
            raise KeyError(f"The specified data user name '{name}' does not exist.")
        return self._data_users_dict[name]

    def setup(self) -> None:
        """Setup procedure to be performed before training starts."""

        # 他のTrainerでモデルの学習可能状態が変えられている可能性があるため、再設定する。
        # Reset the model's trainability as it may have been altered by other Trainers.
        for frozen_model_name in self._frozen_model_names:
            self.make_model_frozen(self._get_model(frozen_model_name))

        for training_model_name in self._training_model_names:
            self.make_model_trainable(self._get_model(training_model_name))

    @abstractmethod
    def train(self) -> None:
        """Train the deep neural network models.

        Please build the models, optimizers, dataset, and other components in this method.
        This method is called repeatedly.

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
        model_wrapper = self._model_wrappers_dict[name]
        inference_wrapper = self._inference_wrappers_dict[name]

        # 学習されたモデルを推論用にセットアップ。
        self.make_model_frozen(model_wrapper)

        # 学習モデルと推論モデルをスイッチ。
        model_wrapper.model, inference_wrapper.model = inference_wrapper.model, model_wrapper.model

        # model_wrapperには推論に使われていた古いモデルが渡されるため、パラメータを同期。
        model_wrapper.model.load_state_dict(inference_wrapper.model.state_dict())

        # 推論に使われていたモデルを学習モードにする。
        self.make_model_trainable(model_wrapper)

    def teardown(self) -> None:
        """Teardown procedure to be performed after training."""
        pass

    def run(self) -> None:
        """Runs the training process."""
        self.setup()
        self.train()
        self.synchronize()
        self.teardown()
