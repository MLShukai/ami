from typing import Any

import torch
from typing_extensions import override
from vrchat_io.controller.osc import RESET_VALUES, Axes, Buttons, InputController
from vrchat_io.controller.wrappers.osc import MultiInputWrapper

from .base_actuator import BaseActuator
from .first_order_delay_system import FirstOrderDelaySystemStepResponce

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

    def __init__(
        self,
        osc_address: str = "127.0.0.1",
        osc_sender_port: int = 9000,
        move_vertical_velocity: float = 1.0,
        move_horizontal_velocity: float = 1.0,
        look_horizontal_velocity: float = 1.0,  # > 0.5 to move.
    ) -> None:
        """Create VRChatOSCDiscreteActuator object.

        Args:
            osc_address: IP address of OSC server.
            osc_sender_port: Port number of OSC server.
            move_vertical_velocity: The velocity for moving forward or backward.
            move_horizontal_velocity: The velocity for moving left or right.
            look_horizontal_velocity: The velocity for turning left or right.
        """
        controller = InputController((osc_address, osc_sender_port))
        self.controller = MultiInputWrapper(controller)
        self._previous_action = STOP_ACTION

        self.move_vertical_velocity = move_vertical_velocity
        self.move_horizontal_velocity = move_horizontal_velocity
        self.look_horizontal_velocity = look_horizontal_velocity

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

    def move_vertical_to_command(self, move_vert: int) -> dict[str, float]:
        match move_vert:
            case 0:  # stop
                return {Axes.Vertical: 0.0}
            case 1:  # forward
                return {Axes.Vertical: self.move_vertical_velocity}
            case 2:  # backward
                return {Axes.Vertical: -self.move_vertical_velocity}
            case _:
                raise ValueError(f"Action choices are 0 to 2! Input: {move_vert}")

    def move_horizontal_to_command(self, move_horzn: int) -> dict[str, float]:
        match move_horzn:
            case 0:  # stop
                return {Axes.Horizontal: 0.0}
            case 1:  # forward
                return {Axes.Horizontal: self.move_horizontal_velocity}
            case 2:  # backward
                return {Axes.Horizontal: -self.move_horizontal_velocity}
            case _:
                raise ValueError(f"Action choices are 0 to 2! Input: {move_horzn}")

    def look_horizontal_to_command(self, look_horzn: int) -> dict[str, float]:
        match look_horzn:
            case 0:  # stop
                return {Axes.LookHorizontal: 0.0}
            case 1:
                return {Axes.LookHorizontal: self.look_horizontal_velocity}
            case 2:
                return {Axes.LookHorizontal: -self.look_horizontal_velocity}
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


class FirstOrderDelaySystemDiscreteActuator(BaseActuator[torch.Tensor]):
    """Treats discrete actions for VRChat OSC with first-order delay system
    dynamics.

    This actuator provides smooth acceleration and deceleration for continuous
    movements (vertical, horizontal, and look) by applying first-order delay
    system dynamics. Each continuous action gradually approaches its target
    velocity according to the specified time constant.

    Actions:
        MoveVertical: [0: stop | 1: forward | 2: backward]
            - Smoothly approaches target velocity
            - Forward: +max_move_vertical_velocity
            - Backward: -max_move_vertical_velocity
            - Stop: 0.0

        MoveHorizontal: [0: stop | 1: right | 2: left]
            - Smoothly approaches target velocity
            - Right: +max_move_horizontal_velocity
            - Left: -max_move_horizontal_velocity
            - Stop: 0.0

        LookHorizontal: [0: stop | 1: right | 2: left]
            - Smoothly approaches target velocity
            - Right: +max_look_horizontal_velocity
            - Left: -max_look_horizontal_velocity
            - Stop: 0.0

        Jump: [0: release | 1: do]
            - Discrete action (no smoothing)

        Run: [0: release | 1: do]
            - Discrete action (no smoothing)

    The continuous actions (Move/Look) use first-order delay dynamics:
        τ(dy/dt) + y = u
    where:
        τ: time constant
        y: current velocity
        u: target velocity
    """

    def __init__(
        self,
        osc_address: str = "127.0.0.1",
        osc_sender_port: int = 9000,
        max_move_vertical_velocity: float = 1.0,
        max_move_horizontal_velocity: float = 1.0,
        max_look_horizontal_velocity: float = 1.0,  # > 0.5 to move.
        delta_time: float = 0.1,  # seconds
        time_constant: float = 0.5,  # second
    ) -> None:
        """Create FirstOrderDelaySystemDiscreteActuator object.

        Args:
            osc_address: IP address of OSC server.
            osc_sender_port: Port number of OSC server.
            max_move_vertical_velocity: Maximum velocity for moving forward/backward.
            max_move_horizontal_velocity: Maximum velocity for moving left/right.
            max_look_horizontal_velocity: Maximum velocity for turning left/right.
            delta_time: Time step for numerical integration in seconds.
                Controls how often the velocity updates are calculated.
            time_constant: Time constant for the first-order delay system in seconds.
                Larger values result in slower acceleration/deceleration.
                For example:
                - 0.5s: Reaches ~63% of target velocity in 0.5s
                - 1.0s: Reaches ~63% of target velocity in 1.0s
        """
        controller = InputController((osc_address, osc_sender_port))
        self.controller = MultiInputWrapper(controller)
        self._previous_action = STOP_ACTION.clone()

        self.max_move_vertical_velocity = max_move_vertical_velocity
        self.max_move_horizontal_velocity = max_move_horizontal_velocity
        self.max_look_horizontal_velocity = max_look_horizontal_velocity

        self._move_vertical_system = FirstOrderDelaySystemStepResponce(delta_time, time_constant, 0.0)
        self._move_horizontal_system = FirstOrderDelaySystemStepResponce(delta_time, time_constant, 0.0)
        self._look_horizontal_system = FirstOrderDelaySystemStepResponce(delta_time, time_constant, 0.0)

    def operate(self, action: torch.Tensor) -> None:
        """Operate the actuator with the given action.

        Args:
            action: The command for actuator as a `Tensor`.
        """
        command = self.convert_action_to_command(action.long().tolist())
        self.controller.command(command)
        self._previous_action = action
        self._move_vertical_system.step()
        self._move_horizontal_system.step()
        self._look_horizontal_system.step()

    def setup(self) -> None:
        """Setup the actuator."""
        self.controller.command(RESET_VALUES)

    def teardown(self) -> None:
        """Teardown the actuator."""
        self.controller.command(RESET_VALUES)

    _action_before_on_paused: torch.Tensor  # set in `on_paused`

    @override
    def on_paused(self) -> None:
        self._action_before_on_paused = self._previous_action.clone()
        self.controller.command(RESET_VALUES)

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

    def move_vertical_to_command(self, move_vert: int) -> dict[str, float]:
        match move_vert:
            case 0:  # stop
                self._move_vertical_system.set_target_value(0.0)
            case 1:  # forward
                self._move_vertical_system.set_target_value(self.max_move_vertical_velocity)
            case 2:  # backward
                self._move_vertical_system.set_target_value(-self.max_move_vertical_velocity)
            case _:
                raise ValueError(f"Action choices are 0 to 2! Input: {move_vert}")
        return {Axes.Vertical: self._move_vertical_system.get_current_value()}

    def move_horizontal_to_command(self, move_horzn: int) -> dict[str, float]:
        match move_horzn:
            case 0:  # stop
                self._move_horizontal_system.set_target_value(0.0)
            case 1:  # forward
                self._move_horizontal_system.set_target_value(self.max_move_horizontal_velocity)
            case 2:  # backward
                self._move_horizontal_system.set_target_value(-self.max_move_horizontal_velocity)
            case _:
                raise ValueError(f"Action choices are 0 to 2! Input: {move_horzn}")
        return {Axes.Horizontal: self._move_horizontal_system.get_current_value()}

    def look_horizontal_to_command(self, look_horzn: int) -> dict[str, float]:
        match look_horzn:
            case 0:  # stop
                self._look_horizontal_system.set_target_value(0.0)
            case 1:
                self._look_horizontal_system.set_target_value(self.max_look_horizontal_velocity)
            case 2:
                self._look_horizontal_system.set_target_value(-self.max_look_horizontal_velocity)
            case _:
                raise ValueError(f"Choices are 0 to 2! Input: {look_horzn}")
        return {Axes.LookHorizontal: self._look_horizontal_system.get_current_value()}

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
