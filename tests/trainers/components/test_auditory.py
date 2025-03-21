from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torchaudio

from ami.trainers.components.auditory import IntervalSamplingAudioDataset

N_CHANNELS = 1
ORIGINAL_MAX_SAMPLE_SIZE = 2048
ORIGINAL_SAMPLE_RATE = 24000
OUTPUT_SAMPLE_SIZE = 1280
OUTPUT_SAMPLE_RATE = 16000
NUM_ORIGINAL_AUDIOS = 10
NUM_SELECT = 5
INTERVAL = max(NUM_ORIGINAL_AUDIOS // NUM_SELECT, 1)
ALLCLOSE_ATOL = 1e-4


class TestIntervalSamplingAudioDataset:
    @pytest.fixture
    def original_audios(self) -> list[torch.Tensor]:
        return [
            torch.rand(
                N_CHANNELS,
                torch.randint(1, ORIGINAL_MAX_SAMPLE_SIZE, (1,)),
                dtype=torch.float32,
            )
            for _ in range(NUM_ORIGINAL_AUDIOS)
        ]

    @pytest.fixture
    def expected_audios(self, original_audios) -> list[torch.Tensor]:
        expected_audios: list[torch.Tensor] = []
        for original_audio in original_audios[::INTERVAL][:NUM_SELECT]:
            expected_audio = torchaudio.functional.resample(original_audio, ORIGINAL_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
            # Make length into OUTPUT_SAMPLE_SIZE.
            if expected_audio.size(-1) < OUTPUT_SAMPLE_SIZE:
                expected_audio = nn.functional.pad(
                    expected_audio, pad=[0, OUTPUT_SAMPLE_SIZE - expected_audio.size(-1)], mode="constant", value=0.0
                )
            expected_audio = expected_audio[..., :OUTPUT_SAMPLE_SIZE]
            expected_audios.append(expected_audio)
        return expected_audios

    @pytest.fixture
    def original_audio_dir(self, tmp_path, original_audios):
        audio_dir = tmp_path / "audios"
        audio_dir.mkdir()
        for i, original_audio in enumerate(original_audios):
            torchaudio.save(
                uri=str(audio_dir / f"original_audio_{i}.wav"),
                src=original_audio,
                sample_rate=ORIGINAL_SAMPLE_RATE,
            )
        return audio_dir

    @pytest.fixture
    def dataset(self, original_audio_dir):
        return IntervalSamplingAudioDataset(
            audio_dir=original_audio_dir,
            sample_rate=OUTPUT_SAMPLE_RATE,
            sample_size=OUTPUT_SAMPLE_SIZE,
            num_select=NUM_SELECT,
        )

    def test_init(self, original_audio_dir):
        dataset = IntervalSamplingAudioDataset(
            audio_dir=original_audio_dir,
            sample_rate=OUTPUT_SAMPLE_RATE,
            sample_size=OUTPUT_SAMPLE_SIZE,
            num_select=NUM_SELECT,
        )
        assert len(dataset.audio_files) == NUM_SELECT

    def test_len(self, dataset):
        assert len(dataset) == NUM_SELECT

    def test_getitem(self, dataset, expected_audios):
        assert len(dataset) == len(expected_audios)
        for item, expected_audio in zip(dataset, expected_audios):
            assert isinstance(item, tuple)
            assert len(item) == 1
            assert item[0].shape == expected_audio.shape
            assert torch.allclose(item[0], expected_audio, atol=ALLCLOSE_ATOL)

    def test_pre_loading(self, original_audio_dir, expected_audios):
        dataset = IntervalSamplingAudioDataset(
            audio_dir=original_audio_dir,
            sample_rate=OUTPUT_SAMPLE_RATE,
            sample_size=OUTPUT_SAMPLE_SIZE,
            num_select=NUM_SELECT,
            pre_loading=True,
        )
        assert dataset.audio_data is not None
        assert len(dataset.audio_data) == NUM_SELECT
        assert len(dataset.audio_data) == len(expected_audios)
        assert all(
            torch.allclose(output_audio, expected_audio, atol=ALLCLOSE_ATOL)
            for output_audio, expected_audio in zip(dataset.audio_data, expected_audios)
        )

    def test_without_pre_loading(self, original_audio_dir, expected_audios):
        dataset = IntervalSamplingAudioDataset(
            audio_dir=original_audio_dir,
            sample_rate=OUTPUT_SAMPLE_RATE,
            sample_size=OUTPUT_SAMPLE_SIZE,
            num_select=NUM_SELECT,
            pre_loading=False,
        )
        assert dataset.audio_data is None
        assert len(dataset) == len(expected_audios)
        assert all(torch.allclose(dataset[i][0], expected_audios[i], atol=ALLCLOSE_ATOL) for i in range(len(dataset)))
