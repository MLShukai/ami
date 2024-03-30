import torch
import cv2
from vrchat_io.abc.video_capture import VideoCapture
from vrchat_io.vision import OpenCVVideoCapture
from vrchat_io.vision.wrappers import RatioCropWrapper, ResizeWrapper

from .base_sensor import BaseSensor


class OpenCVImageSensor(BaseSensor[torch.Tensor]):
    camera: VideoCapture

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 512,
        height: int = 512,
        base_fps: float = 60.0,
        bgr2rgb: bool = True,
        aspect_ratio: float | None = None
    ):
        """Create OpenCVImageSensor object.

        Args:
            camera_index: Capture device index.
            width: Width of captured frames.
            height: Height of captured frames.
            base_fps: Base framerate of capture device.
            bgr2rgb: Convert BGR to RGB.
            aspect_ratio: Ratio to crop frames to. If None, use `width/height`.
        """
        camera = OpenCVVideoCapture(
            cv2.VideoCapture(camera_index),
            width=width,
            height=height,
            fps=base_fps,
            bgr2rgb=bgr2rgb
        )

        if aspect_ratio is None:
            aspect_ratio = width / height

        camera = RatioCropWrapper(camera, aspect_ratio)
        self.camera = ResizeWrapper(camera, (width, height))

    def read(self) -> torch.Tensor:
        """Receive observed data and convert dtype and shape of array.

        Returns:
            observation: Tensor formatted for torch DL models.
        """
        return torch.from_numpy(self.camera.read()).clone().permute(2, 0, 1).float() / 255.0
