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
        adjustor: BaseIntervalAdjustor,
    ) -> None:
        super().__init__(environment, agent)
        self.adjustor = adjustor

    def setup(self) -> None:
        self.adjustor.reset()
        super().setup()
        self.adjustor.adjust()

    def step(self) -> None:
        super().step()
        self.adjustor.adjust()
