from pathlib import Path
from typing import Any, Generic

from .._types import ActType, ObsType
from .actuators.base_actuator import BaseActuator
from .base_environment import BaseEnvironment
from .sensors.base_sensor import BaseSensor


class SensorActuatorEnv(BaseEnvironment[ObsType, ActType], Generic[ObsType, ActType]):
    """The environment class that has sensor and actuator."""

    def __init__(self, sensor: BaseSensor[ObsType], actuator: BaseActuator[ActType]) -> None:
        self.sensor = sensor
        self.actuator = actuator

    def observe(self) -> ObsType:
        return self.sensor.read()

    def affect(self, action: ActType) -> None:
        self.actuator.operate(action)

    def setup(self) -> None:
        self.sensor.setup()
        self.actuator.setup()

    def teardown(self) -> None:
        self.sensor.teardown()
        self.actuator.teardown()

    def save_state(self, path: Path) -> None:
        """Saves the internal state to the `path`."""
        path.mkdir()
        self.sensor.save_state(path / "sensor")
        self.actuator.save_state(path / "actuator")
