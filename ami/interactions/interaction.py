from pathlib import Path
from typing import Any

from typing_extensions import override

from ami.checkpointing import SaveAndLoadStateMixin
from ami.threads.thread_control import PauseResumeEventMixin

from ._types import ActType, ObsType
from .agents.base_agent import BaseAgent
from .environments.base_environment import BaseEnvironment
from .io_wrappers.base_io_wrapper import BaseActionWrapper, BaseObservationWrapper


class Interaction(SaveAndLoadStateMixin, PauseResumeEventMixin):
    """The interaction protocol between an environment and an agent."""

    def __init__(
        self,
        environment: BaseEnvironment[ObsType, ActType],
        agent: BaseAgent[ObsType, ActType],
        observation_wrappers: list[BaseObservationWrapper[Any, Any]] | None = None,
        action_wrappers: list[BaseActionWrapper[Any, Any]] | None = None,
    ) -> None:
        """Initializes the interaction with specified environment and agent.

        Args:
            environment: The instance of environment class.
            agent: The instance of agent class.
            observation_wrappers: The list of observation wrappers.
            action_wrappers: The list of action wrappers.
        """
        self.environment = environment
        self.agent = agent

        if observation_wrappers is None:
            observation_wrappers = []
        if action_wrappers is None:
            action_wrappers = []

        self.observation_wrappers = observation_wrappers
        self.action_wrappers = action_wrappers

    def setup(self) -> None:
        """Called at the start of the interaction."""
        for observation_wrapper in self.observation_wrappers:
            observation_wrapper.setup()

        for action_wrapper in self.action_wrappers:
            action_wrapper.setup()

        self.environment.setup()
        initial_obs = self.environment.observe()
        initial_action = self.agent.setup(self.wrap_observation(initial_obs))
        if initial_action is not None:
            self.environment.affect(self.wrap_action(initial_action))

    def step(self) -> None:
        """Executes a single step of interaction.

        This method is called repeatedly by the inference thread.
        """
        obs = self.environment.observe()
        action = self.agent.step(self.wrap_observation(obs))
        self.environment.affect(self.wrap_action(action))

    def teardown(self) -> None:
        """Called at the end of the interaction."""
        final_obs = self.environment.observe()
        final_action = self.agent.teardown(self.wrap_observation(final_obs))
        if final_action is not None:
            self.environment.affect(self.wrap_action(final_action))
        self.environment.teardown()

        for observation_wrapper in self.observation_wrappers:
            observation_wrapper.teardown()

        for action_wrapper in self.action_wrappers:
            action_wrapper.teardown()

    @override
    def save_state(self, path: Path) -> None:
        """Saves the internal state to `path`."""
        path.mkdir()
        self.agent.save_state(path / "agent")
        self.environment.save_state(path / "environment")

        for i, observation_wrapper in enumerate(self.observation_wrappers):
            observation_wrapper.save_state(path / f"observation_wrapper.{i}")

        for i, action_wrapper in enumerate(self.action_wrappers):
            action_wrapper.save_state(path / f"action_wrapper.{i}")

    @override
    def load_state(self, path: Path) -> None:
        """Loads the internal state from the `path`."""
        self.agent.load_state(path / "agent")
        self.environment.load_state(path / "environment")

        for i, observation_wrapper in enumerate(self.observation_wrappers):
            observation_wrapper.load_state(path / f"observation_wrapper.{i}")

        for i, action_wrapper in enumerate(self.action_wrappers):
            action_wrapper.load_state(path / f"action_wrapper.{i}")

    @override
    def on_paused(self) -> None:
        self.environment.on_paused()
        self.agent.on_paused()

        for observation_wrapper in self.observation_wrappers:
            observation_wrapper.on_paused()

        for action_wrapper in self.action_wrappers:
            action_wrapper.on_paused()

    @override
    def on_resumed(self) -> None:
        self.environment.on_resumed()
        self.agent.on_resumed()

        for observation_wrapper in self.observation_wrappers:
            observation_wrapper.on_resumed()

        for action_wrapper in self.action_wrappers:
            action_wrapper.on_resumed()

    def wrap_observation(self, observation: ObsType) -> ObsType:
        """Applies the observation wrappers to observation."""
        for wrapper in self.observation_wrappers:
            observation = wrapper.wrap(observation)
        return observation

    def wrap_action(self, action: ActType) -> ActType:
        for wrapper in self.action_wrappers:
            action = wrapper.wrap(action)
        return action
