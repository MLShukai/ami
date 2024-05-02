import pytest
from pytest_mock import MockerFixture

from ami.interactions.environments.actuators.base_actuator import (
    BaseActuator,
    BaseActuatorWrapper,
)


class ActuatorWrapperImpl(BaseActuatorWrapper[int, int]):
    def wrap_action(self, action: int) -> int:
        return action


class TestActuatorWrapper:
    @pytest.fixture
    def mocked_actuator_wrapper(self, mocker: MockerFixture) -> None:
        return ActuatorWrapperImpl(mocker.Mock(BaseActuator))

    def test_save_and_load_state(self, mocked_actuator_wrapper: ActuatorWrapperImpl, tmp_path):

        actuator_path = tmp_path / "actuator"
        mocked_actuator_wrapper.save_state(actuator_path)
        mocked_actuator_wrapper._actuator.save_state.assert_called_once_with(actuator_path)
        mocked_actuator_wrapper.load_state(actuator_path)
        mocked_actuator_wrapper._actuator.load_state.assert_called_once_with(actuator_path)

    def test_system_event_callbacks(self, mocked_actuator_wrapper: ActuatorWrapperImpl, mocker: MockerFixture) -> None:
        mocked_actuator_wrapper.on_paused()
        mocked_actuator_wrapper._actuator.on_paused.assert_called_once()

        mocked_actuator_wrapper.on_resumed()
        mocked_actuator_wrapper._actuator.on_resumed.assert_called_once()
