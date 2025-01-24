from pathlib import Path

import pytest
import torch
import torchaudio
from torchvision.io import write_jpeg

from ami.trainers.components.auditory import IntervalSamplingAudioDataset

N_CHANNELS = 1
SAMPLE_SIZE = 1024
SAMPLE_RATE = 16000
NUM_TOTAL_AUDIOS = 10
NUM_SELECT = 5


class TestIntervalSamplingAudioDataset:
    @pytest.fixture
    def sample_audio_dir(self, tmp_path):
        audio_dir = tmp_path / "audios"
        audio_dir.mkdir()
        for i in range(NUM_TOTAL_AUDIOS):
            sample_audio = torch.zeros(N_CHANNELS, SAMPLE_SIZE, dtype=torch.float32)
            torchaudio.save(
                uri=str(audio_dir / f"sample_audio_{i}.wav"),
                src=sample_audio,
                sample_rate=SAMPLE_RATE,
            )
        return audio_dir

    @pytest.fixture
    def dataset(self, sample_audio_dir):
        return IntervalSamplingAudioDataset(sample_audio_dir, sample_rate=SAMPLE_RATE, num_select=NUM_SELECT)

    def test_init(self, sample_audio_dir):
        dataset = IntervalSamplingAudioDataset(sample_audio_dir, sample_rate=SAMPLE_RATE, num_select=NUM_SELECT)
        assert len(dataset.audio_files) == NUM_SELECT
        assert dataset._interval == NUM_TOTAL_AUDIOS // NUM_SELECT

    def test_len(self, dataset):
        assert len(dataset) == NUM_SELECT

    def test_getitem(self, dataset):
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 1
        assert item[0].shape == (N_CHANNELS, SAMPLE_SIZE)
        # assert torch.allclose(item[0], torch.zeros(3, 32, 32, dtype=torch.uint8))

    def test_pre_loading(self, sample_audio_dir):
        dataset = IntervalSamplingAudioDataset(
            sample_audio_dir, sample_rate=SAMPLE_RATE, num_select=NUM_SELECT, pre_loading=True
        )
        assert dataset.audio_data is not None
        assert len(dataset.audio_data) == NUM_SELECT
        assert all(audio.shape == (N_CHANNELS, SAMPLE_SIZE) for audio in dataset.audio_data)

    def test_without_pre_loading(self, sample_audio_dir):
        dataset = IntervalSamplingAudioDataset(
            sample_audio_dir, sample_rate=SAMPLE_RATE, num_select=NUM_SELECT, pre_loading=False
        )
        assert dataset.audio_data is None
        assert all(dataset[i][0].shape == (N_CHANNELS, SAMPLE_SIZE) for i in range(len(dataset)))
