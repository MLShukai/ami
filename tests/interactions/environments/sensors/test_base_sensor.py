"""This file contains test cases for ensuring the implemented sensor
behaviors."""
import pytest
from pytest_mock import MockerFixture

from ami.interactions.environments.sensors.base_sensor import (
    BaseSensor,
    BaseSensorWrapper,
)


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

    def test_save_and_load_state(self, sensor_wrapper: IncrementSensorWrapper, mocker: MockerFixture, tmp_path):

        spied_save_state = mocker.spy(sensor_wrapper._sensor, "save_state")
        spied_load_state = mocker.spy(sensor_wrapper._sensor, "load_state")

        sensor_path = tmp_path / "sensor"
        sensor_wrapper.save_state(sensor_path)
        spied_save_state.assert_called_once_with(sensor_path)
        sensor_wrapper.load_state(sensor_path)
        spied_load_state.assert_called_once_with(sensor_path)

    def test_system_event_callbacks(self, sensor_wrapper: IncrementSensorWrapper, mocker: MockerFixture) -> None:
        mock_on_paused = mocker.spy(sensor_wrapper._sensor, "on_paused")
        mock_on_resumed = mocker.spy(sensor_wrapper._sensor, "on_resumed")

        sensor_wrapper.on_paused()
        mock_on_paused.assert_called_once()

        sensor_wrapper.on_resumed()
        mock_on_resumed.assert_called_once()
