"""Video Action Recorder.

This script records video and keyboard actions using VRChatOSCDiscreteActuator.

Additional Dependency:
    pip install keyboard

Usage:
    python video_action_recorder.py [--fps FPS] [--width WIDTH] [--height HEIGHT] [--osc-ip OSC_IP] [--osc-port OSC_PORT] [--device DEVICE]

    Optional arguments:
    --fps FPS           Frames per second for video recording (default: 10)
    --width WIDTH       Width of the video frame (default: 1280)
    --height HEIGHT     Height of the video frame (default: 720)
    --osc-ip OSC_IP     IP address for OSC server (default: 127.0.0.1)
    --osc-port OSC_PORT Port number for OSC server (default: 9000)
    --device DEVICE     Video capture device index (default: 0)

    Press 'q' to stop recording.

NOTE: Linux User
    You need to run this script as root user.
    Because keyboard module requires root user to access keyboard device.
    So run the following command in the project root directory.
    ```sh
    PYTHON_PATH=$(which python3) && sudo $PYTHON_PATH ./scripts/video_action_recorder.py
    ```

Output:
    - Video file: output_YYYY-MM-DD_HH-MM-SS.mp4
    - Action log: action_log_YYYY-MM-DD_HH-MM-SS.csv
"""

import argparse
import csv
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple

import cv2
import keyboard
import torch
from numpy.typing import NDArray

from ami.interactions.environments.actuators.vrchat_osc_discrete_actuator import (
    VRChatOSCDiscreteActuator,
)


class VideoRecorder:
    def __init__(self, output_file: str, fps: int, resolution: Tuple[int, int], device: int):
        self.output_file = output_file
        self.fps = fps
        self.resolution = resolution
        self.device = device
        self.capture: Optional[cv2.VideoCapture] = None
        self.out: Optional[cv2.VideoWriter] = None

    def start_recording(self) -> None:
        self.capture = cv2.VideoCapture(self.device)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.output_file, fourcc, self.fps, self.resolution)

    def record_frame(self) -> Optional[NDArray[Any]]:
        if self.capture is None or self.out is None:
            return None
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.resize(frame, self.resolution)
            self.out.write(frame)
            return frame
        return None

    def stop_recording(self) -> None:
        if self.capture:
            self.capture.release()
        if self.out:
            self.out.release()


class KeyboardActionHandler:
    def __init__(self, actuator: VRChatOSCDiscreteActuator):
        self.actuator = actuator
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
        self.actions = self.initial_actions.clone()

        for key, (action_idx, value) in self.key_map.items():
            if keyboard.is_pressed(key):
                self.actions[action_idx] = value

        self.actuator.operate(self.actions)
        return not keyboard.is_pressed("q")  # Return False if 'q' is pressed to quit

    def get_action_array(self) -> List[int]:
        return self.actions.tolist()


class ActionRecorder:
    def __init__(self, filename: str):
        self.filename = filename
        self.headers = ["Timestamp", "MoveVertical", "MoveHorizontal", "LookHorizontal", "Jump", "Run"]
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        with open(self.filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.headers)

    def record_action(self, action_array: List[int]) -> None:
        timestamp = f"{time.time():.5f}"
        row = [timestamp] + action_array
        with open(self.filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)


def get_timestamp_string() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video and Action Recording Script")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for video recording")
    parser.add_argument("--width", type=int, default=1280, help="Width of the video frame")
    parser.add_argument("--height", type=int, default=720, help="Height of the video frame")
    parser.add_argument("--osc-ip", type=str, default="127.0.0.1", help="IP address for OSC server")
    parser.add_argument("--osc-port", type=int, default=9000, help="Port number for OSC server")
    parser.add_argument("--device", type=int, default=0, help="Video capture device index")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Configuration
    timestamp = get_timestamp_string()
    video_output = f"output_{timestamp}.mp4"
    action_log = f"action_log_{timestamp}.csv"
    resolution = (args.width, args.height)

    # Initialize components
    video_recorder = VideoRecorder(video_output, args.fps, resolution, args.device)
    actuator = VRChatOSCDiscreteActuator(
        osc_address=args.osc_ip,
        osc_sender_port=args.osc_port,
        move_vertical_velocity=1.0,
        move_horizontal_velocity=1.0,
        look_horizontal_velocity=1.0,
    )
    action_handler = KeyboardActionHandler(actuator)
    action_recorder = ActionRecorder(action_log)

    # Start video recording and setup actuator
    video_recorder.start_recording()
    actuator.setup()

    # Main loop
    frame_time = 1 / args.fps
    next_frame_time = time.time()

    print("Recording started. Press 'q' to stop.")

    try:
        while True:
            current_time = time.time()

            if current_time >= next_frame_time:
                # Record video frame
                frame = video_recorder.record_frame()
                if frame is not None:
                    cv2.imshow("Recording", frame)

                # Update actions and send OSC
                if not action_handler.update():
                    break

                # Record actions
                action_array = action_handler.get_action_array()
                action_recorder.record_action(action_array)
                print(f"\r{action_array}", end="")

                # Calculate next frame time
                next_frame_time += frame_time

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Cleanup
        actuator.teardown()
        video_recorder.stop_recording()
        cv2.destroyAllWindows()

        print(f"\nRecording finished. Video saved as {video_output}, actions logged in {action_log}")


if __name__ == "__main__":
    main()
