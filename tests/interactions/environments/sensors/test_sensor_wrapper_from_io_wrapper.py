from pytest_mock import MockerFixture

from ami.interactions.environments.sensors.sensor_wrapper_from_io_wrapper import (
    BaseIOWrapper,
    BaseSensor,
    SensorWrapperFromIOWrapper,
)


class TestSensorWrapperFromIOWrapper:
    def test_wrapper_method_call(self, mocker: MockerFixture, tmp_path):
        mock_io_wrapper = mocker.Mock(BaseIOWrapper)
        mock_io_wrapper.wrap.return_value = 1
        wrapper = SensorWrapperFromIOWrapper(
            sensor=mocker.Mock(BaseSensor),
            io_wrapper=mock_io_wrapper,
        )

        assert wrapper.read() == 1
        assert wrapper.wrap_observation(0) == 1
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
