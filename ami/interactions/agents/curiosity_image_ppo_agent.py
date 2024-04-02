import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from ...data.buffers.buffer_names import BufferNames
from ...data.step_data import DataKeys, StepData
from ...models.forward_dynamics import ForwardDynamics
from ...models.model_names import ModelNames
from ...models.model_wrapper import ThreadSafeInferenceWrapper
from ...models.policy_value_common_net import PolicyValueCommonNet
from .base_agent import BaseAgent


class CuriosityImagePPOAgent(BaseAgent[Tensor, Tensor]):
    """Image input curiosity agent with ppo policy."""

    def __init__(self, initial_hidden: Tensor) -> None:
        """Constructs Agent.

        Args:
            initial_hidden: Initial hidden state for the forward dynamics model.
        """
        super().__init__()

        self.forward_dynamics_hidden_state = initial_hidden

    def on_inference_models_attached(self) -> None:
        super().on_inference_models_attached()
        self.image_encoder: ThreadSafeInferenceWrapper[nn.Module] = self.get_inference_model(ModelNames.IMAGE_ENCODER)
        self.foward_dynamics: ThreadSafeInferenceWrapper[ForwardDynamics] = self.get_inference_model(
            ModelNames.FORWARD_DYNAMICS
        )
        self.policy_value: ThreadSafeInferenceWrapper[PolicyValueCommonNet] = self.get_inference_model(
            ModelNames.POLICY_VALUE
        )

    def on_data_collectors_attached(self) -> None:
        super().on_data_collectors_attached()
        self.image_collector = self.get_data_collector(BufferNames.IMAGE)
        self.forward_dynamics_trajectory_collector = self.get_data_collector(BufferNames.FORWARD_DYNAMICS_TRAJECTORY)
        self.ppo_trajectory_collector = self.get_data_collector(BufferNames.PPO_TRAJECTORY)

    # ------ Interaction Process ------
    predicted_next_embed_observation_dist: Distribution
    forward_dynamics_hidden_state: Tensor
    step_data: StepData

    def setup(self, observation: Tensor) -> Tensor | None:
        super().setup(observation)

        self.step_data = StepData()

        embed_obs: Tensor = self.image_encoder(observation)
        self.step_data[DataKeys.OBSERVATION] = observation
        self.step_data[DataKeys.EMBED_OBSERVATION] = embed_obs

        self.image_collector.collect(self.step_data)

        action_dist, value = self.policy_value(observation)
        action = action_dist.sample()
        self.step_data[DataKeys.ACTION] = action
        self.step_data[DataKeys.ACTION_LOG_PROBABILITY] = action_dist.log_prob(action)
        self.step_data[DataKeys.VALUE] = value

        self.step_data[DataKeys.HIDDEN] = self.forward_dynamics_hidden_state

        self.forward_dynamics_trajectory_collector.collect(self.step_data)
        pred, hidden = self.foward_dynamics(embed_obs, self.forward_dynamics_hidden_state, action)
        self.predicted_next_embed_observation_dist = pred
        self.forward_dynamics_hidden_state = hidden

        return action

    def step(self, observation: Tensor) -> Tensor:

        # \phi(o_t) -> z_t
        embed_obs = self.image_encoder(observation)
        reward = -self.predicted_next_embed_observation_dist.log_prob(embed_obs)
        self.step_data[DataKeys.REWARD] = reward

        self.ppo_trajectory_collector.collect(self.step_data)  # o_t, a_t, log p(a_t), v_t, r_{t+1}

        action_dist, value = self.policy_value(observation)

        self.step_data[DataKeys.OBSERVATION] = observation
        self.step_data[DataKeys.EMBED_OBSERVATION] = embed_obs

        self.image_collector.collect(self.step_data)

        action_dist, value = self.policy_value(observation)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)

        self.step_data[DataKeys.ACTION] = action
        self.step_data[DataKeys.ACTION_LOG_PROBABILITY] = action_log_prob
        self.step_data[DataKeys.VALUE] = value
        self.step_data[DataKeys.HIDDEN] = self.forward_dynamics_hidden_state

        self.forward_dynamics_trajectory_collector.collect(self.step_data)

        pred, hidden = self.foward_dynamics(embed_obs, self.forward_dynamics_hidden_state, action)
        self.predicted_next_embed_observation_dist = pred
        self.forward_dynamics_hidden_state = hidden

        return action
