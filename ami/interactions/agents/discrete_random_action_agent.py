import random
from typing import Any

import torch
from torch import Tensor
from typing_extensions import override

from .base_agent import BaseAgent


class DiscreteRandomActionAgent(BaseAgent[Any, Tensor]):
    """An agent that selects random discrete actions.

    This agent chooses random actions from a set of discrete choices for
    each action category. It also implements action repetition, where
    each chosen action is repeated for a random number of steps within a
    specified range.
    """

    def __init__(
        self, action_choices_per_category: list[int], min_action_repeat: int = 1, max_action_repeat: int = 1
    ) -> None:
        """Initialize the DiscreteRandomActionAgent.

        Args:
            action_choices_per_category (list[int]): Number of choices for each action category.
            min_action_repeat (int): Minimum number of times an action is repeated.
            max_action_repeat (int): Maximum number of times an action is repeated.
        """

        assert max_action_repeat >= min_action_repeat > 0
        self.action_choices_per_category = action_choices_per_category
        self.max_action_repeat = max_action_repeat
        self.min_action_repeat = min_action_repeat

    @property
    def num_actions(self) -> int:
        return len(self.action_choices_per_category)

    @override
    def setup(self, observation: Any) -> Tensor | None:
        self.remaining_action_repeat_counts = [0] * self.num_actions
        self.action = [0] * self.num_actions
        return None

    @override
    def step(self, observation: Any) -> Tensor:
        for action_idx in range(self.num_actions):

            if self.remaining_action_repeat_counts[action_idx] == 0:
                action_choice = random.randrange(self.action_choices_per_category[action_idx])
                num_repeat = self.min_action_repeat + random.randrange(
                    self.max_action_repeat - self.min_action_repeat + 1
                )
                self.action[action_idx] = action_choice
                self.remaining_action_repeat_counts[action_idx] = num_repeat

            self.remaining_action_repeat_counts[action_idx] -= 1

        return torch.tensor(self.action, dtype=torch.long)
