import cv2
import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from ami.interactions.environments.sensors.video_recording_wrapper import (
    BaseSensor,
    VideoRecordingWrapper,
)

WIDTH = 480
HEIGHT = 640
CHANNELS = 3


class TestVideoRecordingWrapper:

    READ_IMAGE = torch.ones(CHANNELS, HEIGHT, WIDTH)

    @pytest.fixture
    def mock_image_sensor(self, mocker: MockerFixture):
        mock = mocker.Mock(BaseSensor)
        mock.read.return_value = self.READ_IMAGE.clone()
        return mock

    @pytest.fixture
    def wrapper(self, mock_image_sensor, tmp_path):
        sensor = VideoRecordingWrapper(mock_image_sensor, tmp_path / "video", WIDTH, HEIGHT, 30.0)
        sensor.setup()
        assert sensor.video_writer.isOpened()
        yield sensor
        sensor.teardown()
        assert not sensor.video_writer.isOpened()

    def test_setup_video_writer(self, wrapper: VideoRecordingWrapper):
        assert wrapper.video_path.exists()

    def test_wrap_observation(self, wrapper: VideoRecordingWrapper):

        obs = wrapper.read()
        assert torch.equal(obs, self.READ_IMAGE)

    def test_recorded_video(self, wrapper: VideoRecordingWrapper):
        for _ in range(30):
            wrapper.read()
        wrapper.teardown()

        cap = cv2.VideoCapture(str(wrapper.video_path))

        for _ in range(30):
            ret, frame = cap.read()
            assert ret
            np.testing.assert_almost_equal(np.full((HEIGHT, WIDTH, CHANNELS), 255, dtype=np.uint8), frame, -1)

    def test_on_paused_and_resumed(self, wrapper: VideoRecordingWrapper):
        wrapper.read()
        wrapper.on_paused()
        assert not wrapper.video_writer.isOpened()
        previus_file = wrapper.video_path
        wrapper.on_resumed()
        assert previus_file != wrapper.video_path
        assert wrapper.video_writer.isOpened()
