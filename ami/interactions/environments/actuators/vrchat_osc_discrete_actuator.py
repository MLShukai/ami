from typing import Any

import torch
from typing_extensions import override
from vrchat_io.controller.osc import RESET_VALUES, Buttons, InputController
from vrchat_io.controller.wrappers.osc import MultiInputWrapper

from .base_actuator import BaseActuator

ACTION_CHOICES_PER_CATEGORY = [3, 3, 3, 2, 2]
ACTION_CHOICES_PER_CATEGORY_MASKED_JUMP_RUN = [3, 3, 3, 1, 1]  # 去年のAMIと行動設定を合わせるため。

STOP_ACTION = torch.tensor(
    [
        0,  # stop to move vertical.
        0,  # stop to move horizontal.
        0,  # stop to turn horizontal.
        0,  # stop to jump.
        0,  # stop to run.
    ],
    dtype=torch.long,
)


class VRChatOSCDiscreteActuator(BaseActuator[torch.Tensor]):
    """Treats discrete actions for VRChat OSC.

    Actions:
        MoveVertical: [0: stop | 1: forward | 2: backward]
        MoveHorizontal:[0: stop | 1: right | 2: left]
        LookHorizontal: [0: stop | 1: right | 2: left]
        Jump: [0: release | 1: do]
        Run: [0: release | 1: do]
    """

    def __init__(self, osc_address: str = "127.0.0.1", osc_sender_port: int = 9000) -> None:
        """Create VRChatOSCDiscreteActuator object.

        Args:
            osc_address: IP address of OSC server.
            osc_sender_port: Port number of OSC server.
        """
        controller = InputController((osc_address, osc_sender_port))
        self.controller = MultiInputWrapper(controller)
        self._previous_action = STOP_ACTION

    def operate(self, action: torch.Tensor) -> None:
        """Operate the actuator with the given action.

        Args:
            action: The command for actuator as a `Tensor`.
        """
        command = self.convert_action_to_command(action.long().tolist())
        self.controller.command(command)
        self._previous_action = action

    def setup(self) -> None:
        """Setup the actuator."""
        self.controller.command(self.convert_action_to_command(STOP_ACTION.tolist()))

    def teardown(self) -> None:
        """Teardown the actuator."""
        self.controller.command(RESET_VALUES)

    _action_before_on_paused: torch.Tensor  # set in `on_paused`

    @override
    def on_paused(self) -> None:
        self._action_before_on_paused = self._previous_action.clone()
        self.operate(STOP_ACTION)

    @override
    def on_resumed(self) -> None:
        self.operate(self._action_before_on_paused)

    def convert_action_to_command(self, action: list[int]) -> dict[str, Any]:
        """Convert raw action list to command dictionary."""
        command = {}
        command.update(self.move_vertical_to_command(action[0]))
        command.update(self.move_horizontal_to_command(action[1]))
        command.update(self.look_horizontal_to_command(action[2]))
        command.update(self.jump_to_command(action[3]))
        command.update(self.run_to_command(action[4]))
        return command

    @staticmethod
    def move_vertical_to_command(move_vert: int) -> dict[str, int]:
        base_command = {
            Buttons.MoveForward: 0,
            Buttons.MoveBackward: 0,
        }

        match move_vert:
            case 0:  # stop
                return base_command
            case 1:  # forward
                base_command[Buttons.MoveForward] = 1
                return base_command
            case 2:  # backward
                base_command[Buttons.MoveBackward] = 1
                return base_command
            case _:
                raise ValueError(f"Action choices are 0 to 2! Input: {move_vert}")

    @staticmethod
    def move_horizontal_to_command(move_horzn: int) -> dict[str, int]:
        base_command = {
            Buttons.MoveLeft: 0,
            Buttons.MoveRight: 0,
        }

        match move_horzn:
            case 0:  # stop
                return base_command
            case 1:  # forward
                base_command[Buttons.MoveLeft] = 1
                return base_command
            case 2:  # backward
                base_command[Buttons.MoveRight] = 1
                return base_command
            case _:
                raise ValueError(f"Action choices are 0 to 2! Input: {move_horzn}")

    @staticmethod
    def look_horizontal_to_command(look_horzn: int) -> dict[str, int]:
        base_command = {
            Buttons.LookLeft: 0,
            Buttons.LookRight: 0,
        }

        match look_horzn:
            case 0:  # stop
                return base_command
            case 1:
                base_command[Buttons.LookLeft] = 1
                return base_command
            case 2:
                base_command[Buttons.LookRight] = 1
                return base_command
            case _:
                raise ValueError(f"Choices are 0 to 2! Input: {look_horzn}")

    @staticmethod
    def jump_to_command(jump: int) -> dict[str, int]:
        match jump:
            case 0:
                return {Buttons.Jump: 0}
            case 1:
                return {Buttons.Jump: 1}
            case _:
                raise ValueError(f"Choices are 0 or 1! Input: {jump}")

    @staticmethod
    def run_to_command(run: int) -> dict[str, int]:
        match run:
            case 0:
                return {Buttons.Run: 0}
            case 1:
                return {Buttons.Run: 1}
            case _:
                raise ValueError(f"Choices are 0 or 1! Input: {run}")
