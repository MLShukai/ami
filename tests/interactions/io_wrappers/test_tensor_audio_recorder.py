import numpy as np
import pytest
import soundfile as sf
import torch

from ami.interactions.io_wrappers.tensor_audio_recorder import TensorAudioRecorder

CHANNELS = 2
SAMPLE_SIZE = 16080


class TestTensorAudioRecoder:
    TEST_TENSOR = torch.rand(CHANNELS, SAMPLE_SIZE) * 2 - 1

    @pytest.fixture
    def output_dir(self, tmp_path):
        return tmp_path / "audio_output"

    @pytest.fixture
    def recorder(self, output_dir):
        recoder = TensorAudioRecorder(
            output_dir,
            sample_rate=16000,
            channel_size=CHANNELS,
        )
        recoder.setup()
        yield recoder
        recoder.teardown()

    def test_open_file_writer(self, recorder: TensorAudioRecorder):
        assert recorder.file_path.exists()
        assert not recorder.file_writer.closed

    def test_wrap(self, recorder: TensorAudioRecorder):
        result = recorder.wrap(self.TEST_TENSOR)
        assert result is self.TEST_TENSOR

    def test_recorded_audio(self, recorder: TensorAudioRecorder):
        # Record multiple frames of audio
        for _ in range(5):
            recorder.wrap(self.TEST_TENSOR)
        recorder.teardown()

        # Read the recorded audio file
        waveform, sample_rate = sf.read(recorder.file_path)

        # Verify the recorded waveform
        assert sample_rate == 16000  # Check sample rate
        assert waveform.shape[1] == CHANNELS  # Check channel count
        assert waveform.shape[0] == SAMPLE_SIZE * 5  # Check total samples

        # Convert original tensor to numpy for comparison
        expected_waveform = self.TEST_TENSOR.transpose(0, 1).numpy()
        # Compare first frame of waveform
        np.testing.assert_array_almost_equal(waveform[:SAMPLE_SIZE], expected_waveform)

    def test_sample_slice_recording(self, output_dir):
        slice_size = 8000
        recorder = TensorAudioRecorder(
            output_dir, sample_rate=16000, channel_size=CHANNELS, sample_slice_size=slice_size
        )
        recorder.setup()

        recorder.wrap(self.TEST_TENSOR)
        recorder.teardown()

        # Read the recorded audio file
        waveform, _ = sf.read(recorder.file_path)

        # Verify only slice_size samples were recorded
        assert waveform.shape[0] == slice_size

        # Compare the sliced waveform
        expected_waveform = self.TEST_TENSOR.transpose(0, 1)[:slice_size].numpy()
        np.testing.assert_array_almost_equal(waveform, expected_waveform)

    def test_on_paused_and_resumed(self, recorder: TensorAudioRecorder):
        # Record initial waveform
        recorder.wrap(self.TEST_TENSOR)

        # Pause recording
        recorder.on_paused()
        assert recorder.file_writer.closed
        previous_file = recorder.file_path

        # Resume recording
        recorder.on_resumed()
        assert previous_file != recorder.file_path
        assert not recorder.file_writer.closed

        # Record more waveform
        recorder.wrap(self.TEST_TENSOR)

        # Verify both files exist and contain waveform
        assert previous_file.exists()
        assert recorder.file_path.exists()

        # Check the content of both files
        waveform1, _ = sf.read(previous_file)
        waveform2, _ = sf.read(recorder.file_path)

        assert waveform1.shape[0] == SAMPLE_SIZE
        assert waveform2.shape[0] == SAMPLE_SIZE
