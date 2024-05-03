"""This file contains an abstract base class for all trainers."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeAlias

import torch.nn as nn

from ami.checkpointing import SaveAndLoadStateMixin
from ami.threads import PauseResumeEventMixin

from ..data.interfaces import ThreadSafeDataUser
from ..data.utils import DataUsersDict
from ..models.model_wrapper import ModelWrapper
from ..models.utils import InferenceWrappersDict, ModelWrappersDict

ModelWrapperType: TypeAlias = ModelWrapper[Any]


class BaseTrainer(ABC, SaveAndLoadStateMixin, PauseResumeEventMixin):
    """Abstract base class for all trainers.

    The `run` method is called repeatedly in the training thread.

    Override the following methods:
        - `on_model_wrappers_dict_attached`: To retrieve the models.
        - `on_data_users_dict_attached`: To retrieve the data users.
        - `train`: To implement the training process.
        - `is_trainable`: To determine whether or not the training can be executed.

    DNN models and data buffers become available after the thread has started.

    ExampleCode:
        ```py
        class VAETrainer(BaseTrainer):

            def __init__(
                    self,
                    device:torch.device = torch.device("cuda:0"),
                    batch_size:int = 16,
                    lr:float = 0.001
            ) -> None:
                self.device = device
                self.batch_size = batch_size
                self.lr = lr

            def on_model_wrappers_dict_attached(self) -> None:
                self.encoder = self.get_training_model("frame_encoder")
                self.decoder = self.get_training_model("frame_decoder")

            def on_data_users_dict_attached(self) -> None:
                self.data_user = self.get_data_user("frame_buffer")

            def is_trainable(self) -> bool:
                self.data_user.update()
                return len(self.data_user.buffer) >= self.batch_size

            def train(self) -> None:
                # ... getting dataset.
                # ... configure optimizer.
                # ... send models to computing device.
                # ... training loop.
        ```
    """

    _model_wrappers_dict: ModelWrappersDict
    _data_users_dict: DataUsersDict

    def __init__(self) -> None:
        """Constructs the trainer class."""
        super().__init__()
        self._synchronized_model_names: set[str] = set()
        self._training_model_names: set[str] = set()
        self._frozen_model_names: set[str] = set()

    def attach_model_wrappers_dict(
        self,
        model_wrappers_dict: ModelWrappersDict,
    ) -> None:
        """Attaches the model wrappers dict to this trainer."""
        self._model_wrappers_dict = model_wrappers_dict
        self.on_model_wrappers_dict_attached()

    @property
    def _inference_wrappers_dict(self) -> InferenceWrappersDict:
        """Retrieves the `inference_wrappers_dict` from the model wrappers
        dictionary."""
        return self._model_wrappers_dict.inference_wrappers_dict

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

    def _get_model(self, name: str) -> ModelWrapperType:
        """Retrieves the model from the models dictionary by `name`."""
        if name not in self._model_wrappers_dict:
            raise KeyError(f"The specified model name '{name}' does not exist.")
        return self._model_wrappers_dict[name]

    def get_frozen_model(self, name: str) -> ModelWrapperType:
        """Retrieves the parameter-frozen (with `requires_grad=False`) model
        used for inference in the training flow."""
        model = self._get_model(name)
        if name in self._training_model_names:
            raise RuntimeError(f"Model '{name}' is already used as training model!")
        self._frozen_model_names.add(name)
        model.freeze_model()
        return model

    def get_training_model(self, name: str) -> ModelWrapperType:
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

        model.unfreeze_model()
        return model

    def get_data_user(self, name: str) -> ThreadSafeDataUser[Any]:
        """Retrieves the specified data user."""
        if name not in self._data_users_dict:
            raise KeyError(f"The specified data user name '{name}' does not exist.")
        return self._data_users_dict[name]

    def is_trainable(self) -> bool:
        """Determines if the training can be executed.

        This method checks if the training process is currently
        feasible. If it returns `False`, the training procedure is
        skipped. Subclasses should override this method to implement
        custom logic for determining trainability status.
        """
        return True

    def setup(self) -> None:
        """Setup procedure to be performed before training starts."""

        # 他のTrainerでモデルの学習可能状態が変えられている可能性があるため、再設定する。
        # Reset the model's trainability as it may have been altered by other Trainers.
        for frozen_model_name in self._frozen_model_names:
            self._get_model(frozen_model_name).freeze_model()

        for training_model_name in self._training_model_names:
            self._get_model(training_model_name).unfreeze_model()

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
        model_wrapper.freeze_model()

        # 学習モデルと推論モデルをスイッチ。
        model_wrapper.model, inference_wrapper.model = inference_wrapper.model, model_wrapper.model

        # model_wrapperには推論に使われていた古いモデルが渡されるため、パラメータを同期。
        model_wrapper.model.load_state_dict(inference_wrapper.model.state_dict())

        # 推論に使われていたモデルを学習モードにする。
        model_wrapper.unfreeze_model()

    def teardown(self) -> None:
        """Teardown procedure to be performed after training."""
        pass

    def run(self) -> None:
        """Runs the training process."""
        self.setup()
        self.train()
        self.synchronize()
        self.teardown()
