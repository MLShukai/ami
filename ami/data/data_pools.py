"""This file contains all names (keys) of data, base data pool and its
implementations."""
import copy
from abc import ABC, abstractmethod
from collections import UserDict
from enum import StrEnum
from typing import Any, Self


class DataKeys(StrEnum):
    """Enumerates all the names of data obtained from the interaction between
    the environment and the agent."""

    PREVIOUS_ACTION = "previous_action"  # a_{t-1}
    OBSERVATION = "observation"  # o_t
    EMBED_OBSERVATION = "embed_observation"  # z_t
    ACTION = "action"  # a_t
    ACTION_LOG_PROBABILITY = "action_log_probability"
    VALUE = "value"  # v_t
    PREDICTED_NEXT_EMBED_OBSERVATION = "predicted_next_embed_observation"  # \hat{z}_{t+1}
    NEXT_OBSERVATION = "next_observation"  # o_{t+1}
    NEXT_EMBED_OBSERVATION = "next_embed_observation"  # z_{t+1}
    REWARD = "reward"  # r_{t+1}
    NEXT_VALUE = "next_value"  # v_{t+1}


class StepData(UserDict[DataKeys, Any]):
    """Dictionary that holds the data obtained from one step of the agent."""

    def copy(self) -> Self:
        """Return deep copied Self.

        Returns:
            self: deep copied data.
        """
        return copy.deepcopy(self)
