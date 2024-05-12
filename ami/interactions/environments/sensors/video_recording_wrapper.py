import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from torch import Tensor

from ami.interactions.environments.sensors.base_sensor import BaseSensor
from ami.logger import get_inference_thread_logger

from .base_sensor import BaseSensorWrapper


class VideoRecordingWrapper(BaseSensorWrapper[Tensor, Tensor]):
    """Records the image sensor data to file."""

    def __init__(
        self,
        sensor: BaseSensor[Tensor],
        output_dir: str,
        width: int,
        height: int,
        frame_rate: float,
        file_name_format: str = "%Y-%m-%d_%H-%M-%S.%f.mp4",
        fourcc: str = "mp4v",
        do_rgb_to_bgr: bool = True,
        do_scale_255: bool = True,
    ) -> None:
        """Constructs the sensor wrapper.

        Args:
            sensor: An instance of the image sensor.
            output_dir: Path to the directory where the video will be saved.
            width: Width of the image to be recorded.
            height: Height of the image to be recorded.
            frame_rate: Frame rate at which the video is recorded.
            fourcc: FourCC (Four Character Code) used for the video writer.
            file_name_format: File name format that includes a datetime format string.
            do_rgb_to_bgr: Whether to convert RGB format tensors to BGR format.
            do_scale_255: Whether to scale values by 255.
        """

        super().__init__(sensor)

        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir)
        self.frame_size = (width, height)
        self.frame_rate = frame_rate
        self.file_name_format = file_name_format
        self.fourcc = fourcc
        self.do_rgb_to_bgr = do_rgb_to_bgr
        self.do_scale_255 = do_scale_255
        self.logger = get_inference_thread_logger(self.__class__.__name__)

    def setup_video_writer(self) -> None:
        codec = cv2.VideoWriter.fourcc(*self.fourcc)
        self.video_path = self.output_dir / datetime.now().strftime(self.file_name_format)
        self.video_writer = cv2.VideoWriter(str(self.video_path), codec, self.frame_rate, self.frame_size)
        self.logger.info(f"Recording video to '{self.video_path}'")

    def release_video_writer(self) -> None:
        self.logger.info(f"Saved video to '{self.video_path}'")
        self.video_writer.release()

    def setup(self) -> None:
        super().setup()
        self.setup_video_writer()

    def wrap_observation(self, observation: Tensor) -> Tensor:
        """Write image observation to file.

        Observation shape is (C, H, W).
        """
        frame = observation.detach().cpu().clone().numpy()

        frame = np.moveaxis(frame, 0, -1)
        if self.do_scale_255:
            frame = frame * 255

        frame = frame.astype(np.uint8)

        if self.do_rgb_to_bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        self.video_writer.write(frame)

        return observation

    def teardown(self) -> None:
        super().teardown()
        self.release_video_writer()

    def on_paused(self) -> None:
        super().on_paused()
        self.release_video_writer()  # 経験が途切れるため一度リリース

    def on_resumed(self) -> None:
        super().on_resumed()
        self.setup_video_writer()
