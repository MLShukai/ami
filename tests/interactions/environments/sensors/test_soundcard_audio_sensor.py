import shutil
import sys
import time

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

if sys.platform == "linux":
    if shutil.which("pipewire") is None and shutil.which("pulseaudio") is None:
        pytest.skip("Linux audio backend system (pipewire or pulseaudio) is not available.", allow_module_level=True)

from ami.interactions.environments.sensors.soundcard_audio_sensor import (
    SoundcardAudioSensor,
)


class TestSoundcardAudioSensor:
    @pytest.fixture
    def sensor(self, mocker: MockerFixture):
        # Mock SoundcardAudioCapture
        capture = mocker.patch("ami.interactions.environments.sensors.soundcard_audio_sensor.SoundcardAudioCapture")
        # Return zeros array with shape matching block_size and channel_size
        capture().read.return_value = np.zeros((1600, 2), dtype=np.float32)
        return SoundcardAudioSensor(
            device_name="test_device",
            sample_rate=16000,
            channel_size=2,
            read_frame_size=1600,
            block_size=1600,
            dtype=torch.float32,
        )

    def test_read(self, sensor: SoundcardAudioSensor):
        # Raise error before `setup`
        with pytest.raises(RuntimeError):
            sensor.read()

        # Setup and start the sensor
        sensor.setup()

        # Give some time for the background thread to capture data
        time.sleep(0.1)

        # Read audio data
        audio = sensor.read()

        # Verify shape and values
        assert audio.shape == (2, 1600)  # (channel_size, read_frame_size)
        assert torch.equal(audio, torch.zeros(2, 1600))

        # Cleanup
        sensor.teardown()

    def test_initialization_parameters(self):
        # Test invalid parameters
        with pytest.raises(AssertionError):
            SoundcardAudioSensor(read_frame_size=0)

        with pytest.raises(AssertionError):
            SoundcardAudioSensor(channel_size=0)

        with pytest.raises(AssertionError):
            SoundcardAudioSensor(sample_rate=0)

        with pytest.raises(AssertionError):
            SoundcardAudioSensor(block_size=0)

        with pytest.raises(AssertionError):
            SoundcardAudioSensor(read_frame_size=10, block_size=100)

    def test_teardown(self, sensor: SoundcardAudioSensor):
        sensor.setup()
        sensor.teardown()

        # Verify that the background thread has stopped
        assert not sensor._capture_thread.is_alive()
