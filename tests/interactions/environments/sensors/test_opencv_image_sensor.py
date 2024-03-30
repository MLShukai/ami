import pytest

from ami.interactions.environments.sensors.opencv_image_sensor import (
    OpenCVImageSensor,
)


class TestOpenCVImageSensor:
    @pytest.fixture
    def sensor(self):
        return OpenCVImageSensor()

    def test_read(self, sensor):
        sensor.read()
        assert True
