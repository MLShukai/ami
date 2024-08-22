import cv2
import numpy as np
import pytest
import torch

from ami.interactions.io_wrappers.tensor_video_recorder import TensorVideoRecorder

WIDTH = 480
HEIGHT = 640
CHANNELS = 3


class TestTensorVideoRecorder:

    TEST_TENSOR = torch.ones(CHANNELS, HEIGHT, WIDTH)

    @pytest.fixture
    def output_dir(self, tmp_path):
        return tmp_path / "video_output"

    @pytest.fixture
    def recorder(self, output_dir):
        recorder = TensorVideoRecorder(
            str(output_dir),
            WIDTH,
            HEIGHT,
            frame_rate=30.0,
        )
        recorder.setup()
        yield recorder
        recorder.teardown()

    def test_setup_video_writer(self, recorder: TensorVideoRecorder):
        assert recorder.video_path.exists()
        assert recorder.video_writer.isOpened()

    def test_wrap(self, recorder: TensorVideoRecorder):
        result = recorder.wrap(self.TEST_TENSOR)
        assert torch.equal(result, self.TEST_TENSOR)

    def test_recorded_video(self, recorder: TensorVideoRecorder):
        for _ in range(30):
            recorder.wrap(self.TEST_TENSOR)
        recorder.teardown()

        cap = cv2.VideoCapture(str(recorder.video_path))

        for _ in range(30):
            ret, frame = cap.read()
            assert ret
            np.testing.assert_almost_equal(np.full((HEIGHT, WIDTH, CHANNELS), 255, dtype=np.uint8), frame, -1)

    def test_on_paused_and_resumed(self, recorder: TensorVideoRecorder):
        recorder.wrap(self.TEST_TENSOR)
        recorder.on_paused()
        assert not recorder.video_writer.isOpened()
        previous_file = recorder.video_path
        recorder.on_resumed()
        assert previous_file != recorder.video_path
        assert recorder.video_writer.isOpened()
