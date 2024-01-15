from typing import Any

from .actuators.base_actuator import BaseActuator
from .base_environment import BaseEnvironment
from .sensors.base_sensor import BaseSensor


class SensorActuatorEnv(BaseEnvironment):
    """The environment class that has sensor and actuator."""

    def __init__(self, sensor: BaseSensor, actuator: BaseActuator) -> None:
        self.sensor = sensor
        self.actuator = actuator

    def observe(self) -> Any:
        return self.sensor.read()

    def affect(self, action: Any) -> None:
        self.actuator.operate(action)

    def setup(self) -> None:
        self.sensor.setup()
        self.actuator.setup()

    def teardown(self) -> None:
        self.sensor.teardown()
        self.actuator.teardown()
