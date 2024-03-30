import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from ami.interactions.environments.sensors.opencv_image_sensor import OpenCVImageSensor


class TestOpenCVImageSensor:
    @pytest.fixture
    def sensor(self, mocker: MockerFixture):
        cap = mocker.patch("cv2.VideoCapture")
        cap().read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        return OpenCVImageSensor(width=512, height=512)

    def test_read(self, sensor: OpenCVImageSensor):
        image = sensor.read()
        assert image.shape == (3, 512, 512)
        assert torch.equal(image, torch.zeros(3, 512, 512))
