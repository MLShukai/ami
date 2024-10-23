"""First Order Delay System Demo.

This script demonstrates keyboard control with FirstOrderDelaySystemDiscreteActuator,
which provides smooth acceleration and deceleration for continuous movements.

Additional Dependency:
    pip install keyboard

Usage:
    python first_order_delay_system_discrete_actuator_demo.py [--osc-ip OSC_IP] [--osc-port OSC_PORT]
        [--time-constant TIME_CONSTANT] [--delta-time DELTA_TIME] [--max-velocity MAX_VELOCITY]

    Optional arguments:
    --osc-ip OSC_IP           IP address for OSC server (default: 127.0.0.1)
    --osc-port OSC_PORT       Port number for OSC server (default: 9000)
    --time-constant TIME_CONST Time constant for delay system in seconds (default: 0.5)
    --delta-time DELTA_TIME   Update interval in seconds (default: 0.1)
    --max-velocity MAX_VEL    Maximum velocity for all movements (default: 1.0)

Controls:
    - W/S: Move forward/backward
    - A/D: Move left/right
    - ,/.: Turn left/right
    - Space: Jump
    - Shift: Run
    - Q: Quit

NOTE: Linux User
    You need to run this script as root user.
    Because keyboard module requires root user to access keyboard device.
    So run the following command in the project root directory.
    ```sh
    PYTHON_PATH=$(which python3) && sudo $PYTHON_PATH ./scripts/first_order_delay_system_discrete_actuator_demo.py
    ```

Features:
    - Smooth acceleration and deceleration for all continuous movements
    - Configurable time constant affects how quickly movements reach target velocity
    - Real-time keyboard control with immediate response
"""

import argparse
import sys
import time

import keyboard
import torch

from ami.interactions.environments.actuators.vrchat_osc_discrete_actuator import (
    FirstOrderDelaySystemDiscreteActuator,
)


class KeyboardActionHandler:
    """Handles keyboard input and converts it to discrete actions."""

    def __init__(self) -> None:
        """Initialize the keyboard action handler."""
        self.initial_actions = torch.zeros(5, dtype=torch.long)  # [vertical, horizontal, look, jump, run]
        self.actions = self.initial_actions.clone()

        self.key_map = {  # (action_index, value)
            "w": (0, 1),  # vertical forward
            "s": (0, 2),  # vertical backward
            "d": (1, 1),  # horizontal right
            "a": (1, 2),  # horizontal left
            ".": (2, 1),  # look right
            ",": (2, 2),  # look left
            "space": (3, 1),  # jump
            "shift": (4, 1),  # run
        }

    def update(self) -> bool:
        """Update action state based on current keyboard input.

        Returns:
            bool: False if quit is requested (Q pressed), True otherwise
        """
        self.actions = self.initial_actions.clone()

        for key, (action_idx, value) in self.key_map.items():
            if keyboard.is_pressed(key):
                self.actions[action_idx] = value

        return not keyboard.is_pressed("q")  # Return False if 'q' is pressed to quit

    def get_actions(self) -> torch.Tensor:
        """Get current action state.

        Returns:
            torch.Tensor: Current action state
        """
        return self.actions.clone()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="First Order Delay System Demo")
    parser.add_argument("--osc-ip", type=str, default="127.0.0.1", help="IP address for OSC server")
    parser.add_argument("--osc-port", type=int, default=9000, help="Port number for OSC server")
    parser.add_argument("--time-constant", type=float, default=0.5, help="Time constant for delay system in seconds")
    parser.add_argument("--delta-time", type=float, default=0.1, help="Update interval in seconds")
    parser.add_argument("--max-velocity", type=float, default=1.0, help="Maximum velocity for all movements")
    return parser.parse_args()


def print_status(actions: torch.Tensor, clear: bool = True) -> None:
    """Print current action state to console.

    Args:
        actions: Current action tensor
        clear: Whether to clear the line before printing
    """
    status = [
        "↑" if actions[0] == 1 else "↓" if actions[0] == 2 else "-",  # vertical
        "→" if actions[1] == 1 else "←" if actions[1] == 2 else "-",  # horizontal
        "»" if actions[2] == 1 else "«" if actions[2] == 2 else "-",  # look
        "⋀" if actions[3] == 1 else "-",  # jump
        "⚡" if actions[4] == 1 else "-",  # run
    ]
    print(f"\r[{' '.join(status)}]", end="", flush=True)


def main() -> None:
    """Main function for the demo script."""
    args = parse_arguments()

    # Initialize components
    action_handler = KeyboardActionHandler()
    actuator = FirstOrderDelaySystemDiscreteActuator(
        osc_address=args.osc_ip,
        osc_sender_port=args.osc_port,
        max_move_vertical_velocity=args.max_velocity,
        max_move_horizontal_velocity=args.max_velocity,
        max_look_horizontal_velocity=args.max_velocity,
        delta_time=args.delta_time,
        time_constant=args.time_constant,
    )

    # Setup actuator
    try:
        actuator.setup()
        print("Demo started with:")
        print(f"- Time constant: {args.time_constant}s")
        print(f"- Update interval: {args.delta_time}s")
        print(f"- Max velocity: {args.max_velocity}")
        print("Use WASD, <>, Space, Shift to control. Press Q to quit.")
        print("\nCurrent state:")

        # Main loop
        next_update_time = time.time()
        while True:
            current_time = time.time()

            if current_time >= next_update_time:
                # Update and apply actions
                if not action_handler.update():
                    break

                actions = action_handler.get_actions()
                actuator.operate(actions)
                print_status(actions)

                # Schedule next update
                next_update_time += args.delta_time

            # Small sleep to prevent CPU overload
            time.sleep(max(0, next_update_time - time.time()))

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}", file=sys.stderr)
        raise
    finally:
        # Cleanup
        actuator.teardown()
        print("\nDemo finished.")


if __name__ == "__main__":
    main()
