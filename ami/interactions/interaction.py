from pathlib import Path

from ._types import ActType, ObsType
from .agents.base_agent import BaseAgent
from .environments.base_environment import BaseEnvironment


class Interaction:
    """The interaction protocol between an environment and an agent."""

    def __init__(self, environment: BaseEnvironment[ObsType, ActType], agent: BaseAgent[ObsType, ActType]) -> None:
        """Initializes the interaction with specified environment and agent."""
        self.environment = environment
        self.agent = agent

    def setup(self) -> None:
        """Called at the start of the interaction."""
        self.environment.setup()
        initial_obs = self.environment.observe()
        initial_action = self.agent.setup(initial_obs)
        if initial_action is not None:
            self.environment.affect(initial_action)

    def step(self) -> None:
        """Executes a single step of interaction.

        This method is called repeatedly by the inference thread.
        """
        obs = self.environment.observe()
        action = self.agent.step(obs)
        self.environment.affect(action)

    def teardown(self) -> None:
        """Called at the end of the interaction."""
        final_obs = self.environment.observe()
        final_action = self.agent.teardown(final_obs)
        if final_action is not None:
            self.environment.affect(final_action)
        self.environment.teardown()

    def save_state(self, path: Path) -> None:
        """Saves the internal state to `path`."""
        path.mkdir()
        self.agent.save_state(path / "agent")
        self.environment.save_state(path / "environment")

    def load_state(self, path: Path) -> None:
        """Loads the internal state from the `path`."""
        self.agent.load_state(path / "agent")
        self.environment.load_state(path / "environment")
