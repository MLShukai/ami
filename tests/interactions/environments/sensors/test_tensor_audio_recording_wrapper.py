from pathlib import Path

import torch
from pytest_mock import MockerFixture

from ami.interactions.environments.sensors.tensor_audio_recording_wrapper import (
    BaseSensor,
    TensorAudioRecordingWrapper,
)


class TestTensorAudioRecordingWrapper:
    AUDIO = torch.zeros(2, 16080)

    def test_recording(self, mocker: MockerFixture, tmp_path: Path):
        output_dir = tmp_path / "audio_recordings"
        wrapper = TensorAudioRecordingWrapper(
            sensor=mocker.Mock(BaseSensor),
            output_dir=output_dir,
            sample_rate=16000,
            channel_size=2,
        )

        wrapper.setup()
        assert len(list(output_dir.glob("*.wav"))) == 1
        assert wrapper.wrap_observation(self.AUDIO) is self.AUDIO
        wrapper.on_paused()
        wrapper.on_resumed()
        assert len(list(output_dir.glob("*.wav"))) == 2
        wrapper.teardown()
