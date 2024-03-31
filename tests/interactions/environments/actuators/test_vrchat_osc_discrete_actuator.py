import pytest
import torch
from pytest_mock import MockerFixture
from vrchat_io.controller.osc import Buttons

from ami.interactions.environments.actuators.vrchat_osc_discrete_actuator import (
    VRChatOSCDiscreteActuator,
)


class TestVRChatOSCDiscreteActuator:
    @pytest.fixture
    def actuator(self, mocker: MockerFixture) -> VRChatOSCDiscreteActuator:
        mocker.patch("pythonosc.udp_client.SimpleUDPClient")

        return VRChatOSCDiscreteActuator()

    def test_operate(self, actuator: VRChatOSCDiscreteActuator) -> None:
        actuator.setup()

        actuator.operate(torch.tensor([1, 1, 1, 1, 1], dtype=torch.long))

        with pytest.raises(ValueError):
            actuator.operate(torch.tensor([3, 0, 0, 0, 0], dtype=torch.long))

        actuator.teardown()

    @pytest.mark.parametrize(
        "action, expected_command",
        [
            (
                [0, 0, 0, 0, 0],
                {
                    Buttons.MoveForward: 0,
                    Buttons.MoveBackward: 0,
                    Buttons.MoveLeft: 0,
                    Buttons.MoveRight: 0,
                    Buttons.LookLeft: 0,
                    Buttons.LookRight: 0,
                    Buttons.Jump: 0,
                    Buttons.Run: 0,
                },
            ),
            (
                [1, 1, 1, 1, 1],
                {
                    Buttons.MoveForward: 1,
                    Buttons.MoveBackward: 0,
                    Buttons.MoveLeft: 1,
                    Buttons.MoveRight: 0,
                    Buttons.LookLeft: 1,
                    Buttons.LookRight: 0,
                    Buttons.Jump: 1,
                    Buttons.Run: 1,
                },
            ),
            (
                [2, 2, 2, 0, 0],
                {
                    Buttons.MoveForward: 0,
                    Buttons.MoveBackward: 1,
                    Buttons.MoveLeft: 0,
                    Buttons.MoveRight: 1,
                    Buttons.LookLeft: 0,
                    Buttons.LookRight: 1,
                    Buttons.Jump: 0,
                    Buttons.Run: 0,
                },
            ),
        ],
    )
    def test_convert_action_to_command(self, actuator, action, expected_command):
        assert actuator.convert_action_to_command(action) == expected_command
