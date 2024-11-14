import pytest
import torch
from pytest_mock import MockerFixture
from vrchat_io.controller.osc import Axes, Buttons

from ami.interactions.environments.actuators.vrchat_osc_discrete_actuator import (
    STOP_ACTION,
    VRChatOSCDiscreteActuator,
)


class TestVRChatOSCDiscreteActuator:
    @pytest.fixture
    def actuator(self, mocker: MockerFixture):
        mocker.patch("pythonosc.udp_client.SimpleUDPClient")

        actuator = VRChatOSCDiscreteActuator(
            osc_sender_port=65349,
            move_vertical_velocity=1.0,
            move_horizontal_velocity=0.8,
            look_horizontal_velocity=0.5,
        )
        actuator.setup()
        yield actuator
        actuator.teardown()

    def test_operate(self, actuator: VRChatOSCDiscreteActuator) -> None:
        actuator.operate(torch.tensor([1, 1, 1, 1, 1], dtype=torch.long))

        with pytest.raises(ValueError):
            actuator.operate(torch.tensor([3, 0, 0, 0, 0], dtype=torch.long))

    @pytest.mark.parametrize(
        "action, expected_command",
        [
            (
                [0, 0, 0, 0, 0],
                {
                    Axes.Vertical: 0.0,
                    Axes.Horizontal: 0.0,
                    Axes.LookHorizontal: 0.0,
                    Buttons.Jump: 0,
                    Buttons.Run: 0,
                },
            ),
            (
                [1, 1, 1, 1, 1],
                {
                    Axes.Vertical: 1.0,  # move_vertical_velocity
                    Axes.Horizontal: 0.8,  # move_horizontal_velocity
                    Axes.LookHorizontal: 0.5,  # look_horizontal_velocity
                    Buttons.Jump: 1,
                    Buttons.Run: 1,
                },
            ),
            (
                [2, 2, 2, 0, 0],
                {
                    Axes.Vertical: -1.0,  # -move_vertical_velocity
                    Axes.Horizontal: -0.8,  # -move_horizontal_velocity
                    Axes.LookHorizontal: -0.5,  # -look_horizontal_velocity
                    Buttons.Jump: 0,
                    Buttons.Run: 0,
                },
            ),
        ],
    )
    def test_convert_action_to_command(self, actuator, action, expected_command):
        assert actuator.convert_action_to_command(action) == expected_command

    def test_custom_velocities(self, mocker: MockerFixture):
        mocker.patch("pythonosc.udp_client.SimpleUDPClient")

        actuator = VRChatOSCDiscreteActuator(
            move_vertical_velocity=2.0,
            move_horizontal_velocity=1.5,
            look_horizontal_velocity=0.7,
        )

        command = actuator.convert_action_to_command([1, 1, 1, 0, 0])
        assert command[Axes.Vertical] == 2.0
        assert command[Axes.Horizontal] == 1.5
        assert command[Axes.LookHorizontal] == 0.7

        command = actuator.convert_action_to_command([2, 2, 2, 0, 0])
        assert command[Axes.Vertical] == -2.0
        assert command[Axes.Horizontal] == -1.5
        assert command[Axes.LookHorizontal] == -0.7

    def test_on_paused_and_resumed(self, actuator: VRChatOSCDiscreteActuator, mocker: MockerFixture):
        mock_operate = mocker.spy(actuator, "operate")

        action = torch.tensor([1, 1, 1, 1, 1])
        actuator.operate(action)
        assert torch.equal(mock_operate.call_args.args[0], action)
        actuator.on_paused()
        assert torch.equal(mock_operate.call_args.args[0], STOP_ACTION)
        actuator.on_resumed()
        assert torch.equal(mock_operate.call_args.args[0], action)
