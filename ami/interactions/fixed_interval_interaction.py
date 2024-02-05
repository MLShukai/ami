from ._types import ActType, ObsType
from .agents.base_agent import BaseAgent
from .environments.base_environment import BaseEnvironment
from .interaction import Interaction
from .interval_adjustors import BaseIntervalAdjustor


class FixedIntervalInteraction(Interaction):
    """Executes each interaction step at a fixed interval."""

    def __init__(
        self,
        environment: BaseEnvironment[ObsType, ActType],
        agent: BaseAgent[ObsType, ActType],
        interval_adjustor: BaseIntervalAdjustor,
    ) -> None:
        super().__init__(environment, agent)
        self.interval_adjustor = interval_adjustor

    def setup(self) -> None:
        self.interval_adjustor.reset()
        super().setup()
        self.interval_adjustor.adjust()  # For acting in setup.

    def step(self) -> None:
        super().step()
        self.interval_adjustor.adjust()
