import pytest
from pytest_mock import MockerFixture

from ami.interactions.environments.sensor_actuator_env import (
    BaseActuator,
    BaseSensor,
    SensorActuatorEnv,
)


class TestSensorActuatorEnv:
    @pytest.fixture
    def environment(self, mocker: MockerFixture) -> SensorActuatorEnv:
        sensor = mocker.Mock(spec=BaseSensor)
        sensor.read.return_value = 0
        actuator = mocker.Mock(spec=BaseActuator)
        return SensorActuatorEnv(sensor, actuator)

    def test_observe(self, environment: SensorActuatorEnv) -> None:
        assert environment.observe() == 0

    def test_affect(self, environment: SensorActuatorEnv) -> None:
        environment.affect(0)
        environment.actuator.operate.assert_called_once_with(0)

    def test_setup(self, environment: SensorActuatorEnv) -> None:
        environment.setup()
        environment.sensor.setup.assert_called_once()
        environment.actuator.setup.assert_called_once()

    def test_teardown(self, environment: SensorActuatorEnv) -> None:
        environment.teardown()
        environment.sensor.teardown.assert_called_once()
        environment.actuator.teardown.assert_called_once()
