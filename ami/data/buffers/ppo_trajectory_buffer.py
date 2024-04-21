from collections import OrderedDict

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from ..step_data import DataKeys
from .causal_data_buffer import CausalDataBuffer


class PPOTrajectoryBuffer(CausalDataBuffer):
    """Buffering the trajectory data for ppo training.

    Advantageの計算で次ステップの価値関数の推定値を用いるため、返されるデータセットの長さは `len - 1`となる。

    Returning objects are;
        - observations
        - actions
        - action log probabilities
        - advantages
        - returns
        - values
    """

    def __init__(self, max_len: int, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        """
        Args:
            max_size: The max size of internal buffer.
            gamma: Discount factor.
            gae_lambda: The lambda of generalized advantage estimation.
        """
        super().__init__(
            max_len,
            key_list=[
                DataKeys.OBSERVATION,
                DataKeys.ACTION,
                DataKeys.ACTION_LOG_PROBABILITY,
                DataKeys.REWARD,
                DataKeys.VALUE,
            ],
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    @property
    def dataset_size(self) -> int:
        return max(len(self) - 1, 0)

    def make_dataset(self) -> TensorDataset:
        tensor_dict = OrderedDict()
        for key in self._key_list:
            tensor_dict[key] = torch.stack(list(self.buffer_dict[key]))

        observations = tensor_dict[DataKeys.OBSERVATION][:-1]
        actions = tensor_dict[DataKeys.ACTION][:-1]
        logprobs = tensor_dict[DataKeys.ACTION_LOG_PROBABILITY][:-1]
        rewards = tensor_dict[DataKeys.REWARD][:-1]

        raw_values = tensor_dict[DataKeys.VALUE]
        final_next_value = raw_values[-1]
        values = raw_values[:-1]

        advantages = compute_advantage(rewards, values, final_next_value, self.gamma, self.gae_lambda)

        next_values = raw_values[1:]
        value_targets = rewards + self.gamma * next_values

        return TensorDataset(observations, actions, logprobs, advantages, value_targets, values)


def compute_advantage(
    rewards: Tensor, values: Tensor, final_next_value: Tensor, gamma: float, gae_lambda: float
) -> Tensor:
    """Compute advantages from values.

    Args:
        rewards: shape (step length, )
        values: shape (step length, )
        final_next_value: shape (1,)
        gamma: Discount factor.
        gae_lambda: The lambda of generalized advantage estimation.

    Returns:
        advantages: shape
    """
    advantages = torch.empty_like(values)

    lastgaelam = torch.Tensor([0.0])

    for t in reversed(range(values.size(0))):
        if t == values.size(0) - 1:
            nextvalues = final_next_value
        else:
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * lastgaelam

    return advantages
