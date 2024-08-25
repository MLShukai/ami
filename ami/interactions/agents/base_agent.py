"""This file contains the abstract base agent class."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic

import torch.nn as nn

from ami.checkpointing import SaveAndLoadStateMixin
from ami.threads.thread_control import PauseResumeEventMixin

from ...data.utils import DataCollectorsDict, ThreadSafeDataCollector
from ...models.utils import InferenceWrappersDict, ThreadSafeInferenceWrapper
from .._types import ActType, ObsType


class BaseAgent(ABC, Generic[ObsType, ActType], SaveAndLoadStateMixin, PauseResumeEventMixin):
    """Abstract base agent class for interacting with the environment.

    Methods to override:
        - `on_inference_models_attached`: Retrieves inference models for the agent.
        - `step`: Processes an observation from the environment and returns an action.

    NOTE: The `data_collectors` attribute is initialized and available before the `setup` method is called.
    """

    _inference_models: InferenceWrappersDict
    data_collectors: DataCollectorsDict

    def attach_inference_models(self, inference_models: InferenceWrappersDict) -> None:
        """Attaches the inference models (wrappers dict) to this agent."""
        self._inference_models = inference_models
        self.on_inference_models_attached()

    def on_inference_models_attached(self) -> None:
        """Callback method for when infernece models are attached to the agent.

        Override this method to retrieve DNN models for inference. Use
        `get_inference_model` to retrieve a model.
        """
        pass

    def attach_data_collectors(self, data_collectors: DataCollectorsDict) -> None:
        """Attaches the data collectors (dict) to this agent."""
        self.data_collectors = data_collectors
        self.on_data_collectors_attached()

    def on_data_collectors_attached(self) -> None:
        """Callback method for when data collectors are attached to this
        agent."""
        pass

    def check_model_exists(self, name: str) -> bool:
        return name in self._inference_models

    def get_inference_model(self, name: str) -> ThreadSafeInferenceWrapper[Any]:
        if not self.check_model_exists(name):
            raise KeyError(f"The specified model name '{name}' does not exist.")
        return self._inference_models[name]

    def get_data_collector(self, name: str) -> ThreadSafeDataCollector[Any]:
        if name not in self.data_collectors:
            raise KeyError(f"The specified data collector name '{name}' does not exist.")
        return self.data_collectors[name]

    @abstractmethod
    def step(self, observation: ObsType) -> ActType:
        """Processes the observation and returns an action to the environment.

        Args:
            observation: Data read from the environment.

        Returns:
            action: Action data intended to affect the environment.
        """
        raise NotImplementedError

    def setup(self, observation: ObsType) -> ActType | None:
        """Setup procedure for the Agent.

        Args:
            observation: Initial observation from the environment.

        Returns:
            action: Initial action to be taken in response to the environment during interaction. Returning no action is also an option.
        """
        return None

    def teardown(self, observation: ObsType) -> ActType | None:
        """Teardown procedure for the Agent.

        Args:
            observation: Final observation from the environment.

        Returns:
            action: Final action to be taken in the interaction. Returning no action is also an option.
        """
        return None
