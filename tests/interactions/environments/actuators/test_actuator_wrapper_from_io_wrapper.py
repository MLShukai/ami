from pytest_mock import MockerFixture

from ami.interactions.environments.actuators.actuator_wrapper_from_io_wrapper import (
    ActuatorWrapperFromIOWrapper,
    BaseActuator,
    BaseIOWrapper,
)


class TestActuatorWrapperFromIOWrapper:
    def test_wrapper_method_call(self, mocker: MockerFixture, tmp_path):
        mock_io_wrapper = mocker.Mock(BaseIOWrapper)
        mock_io_wrapper.wrap.return_value = 1
        mock_actuator = mocker.Mock(BaseActuator)

        wrapper = ActuatorWrapperFromIOWrapper(
            actuator=mock_actuator,
            io_wrapper=mock_io_wrapper,
        )

        wrapper.operate(0)
        wrapper.setup()
        wrapper.teardown()
        wrapper.on_paused()
        wrapper.on_resumed()
        wrapper.save_state(tmp_path)
        wrapper.load_state(tmp_path)

        mock_io_wrapper.wrap.assert_called_with(0)
        mock_io_wrapper.setup.assert_called_once()
        mock_io_wrapper.teardown.assert_called_once()
        mock_io_wrapper.on_paused.assert_called_once()
        mock_io_wrapper.on_resumed.assert_called_once()
        mock_io_wrapper.save_state.assert_called_once_with(tmp_path)
        mock_io_wrapper.load_state.assert_called_once_with(tmp_path)

        mock_actuator.operate.assert_called_with(1)
        mock_actuator.setup.assert_called_once()
        mock_actuator.teardown.assert_called_once()
        mock_actuator.on_paused.assert_called_once()
        mock_actuator.on_resumed.assert_called_once()
        mock_actuator.save_state.assert_called_once_with(tmp_path)
        mock_actuator.load_state.assert_called_once_with(tmp_path)
