from .base_interaction import BaseInteraction


class SimpleInteraction(BaseInteraction):
    """A simple implementation of the interaction protocol between an
    environment and an agent."""

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
