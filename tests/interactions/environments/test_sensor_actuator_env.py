from pathlib import Path

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

    def test_save_and_load_state(self, environment: SensorActuatorEnv, tmp_path: Path) -> None:
        env_path = tmp_path / "environment"
        environment.save_state(env_path)
        sensor_path = env_path / "sensor"
        actuator_path = env_path / "actuator"
        assert env_path.exists()

        environment.sensor.save_state.assert_called_once_with(sensor_path)
        environment.actuator.save_state.assert_called_once_with(actuator_path)
        assert not sensor_path.exists()
        assert not actuator_path.exists()

        environment.load_state(env_path)
        environment.sensor.load_state.assert_called_once_with(sensor_path)
        environment.actuator.load_state.assert_called_once_with(actuator_path)

    def test_pause_resume_event_callbacks(self, environment: SensorActuatorEnv, mocker: MockerFixture) -> None:
        environment.on_paused()
        environment.sensor.on_paused.assert_called_once()
        environment.actuator.on_paused.assert_called_once()

        environment.on_resumed()
        environment.sensor.on_resumed.assert_called_once()
        environment.actuator.on_resumed.assert_called_once()
