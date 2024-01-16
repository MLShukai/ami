"""This file contains test cases for ensuring the implemented sensor
behaviors."""
import pytest

from ami.environments.sensors.base_sensor import BaseSensor, BaseSensorWrapper


class IntSensorImpl(BaseSensor):
    def read(self) -> int:
        return 0


class IncrementSensorWrapper(BaseSensorWrapper):
    def wrap_observation(self, observation: int) -> int:
        return observation + 1


class TestSensorWrapper:
    @pytest.fixture
    def sensor_wrapper(self) -> IncrementSensorWrapper:
        return IncrementSensorWrapper(IntSensorImpl())

    def test_read(self, sensor_wrapper: BaseSensorWrapper) -> None:
        assert sensor_wrapper.read() == 1
