from typing import Generic

from ._types import ActType, ObsType
from .agents.base_agent import BaseAgent
from .base_interaction import BaseInteraction
from .environments.base_environment import BaseEnvironment


class SimpleInteraction(BaseInteraction, Generic[ActType, ObsType]):
    """A simple implementation of the interaction protocol between an
    environment and an agent."""

    def __init__(self, environment: BaseEnvironment[ObsType, ActType], agent: BaseAgent[ObsType, ActType]) -> None:
        """Initializes the interaction with specified environment and agent."""
        self.environment = environment
        self.agent = agent

    def setup(self) -> None:
        self.environment.setup()
        initial_obs = self.environment.observe()
        initial_action = self.agent.setup(initial_obs)
        if initial_action is not None:
            self.environment.affect(initial_action)

    def step(self) -> None:
        obs = self.environment.observe()
        action = self.agent.step(obs)
        self.environment.affect(action)

    def teardown(self) -> None:
        final_obs = self.environment.observe()
        final_action = self.agent.teardown(final_obs)
        if final_action is not None:
            self.environment.affect(final_action)
        self.environment.teardown()
