import numpy as np
import pytest
import soundfile as sf
import torch

from ami.interactions.environments.audio_observation_generator import (
    AudioFilesObservationGenerator,
    ChunkedStridedAudioReader,
)


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a sample audio file for testing."""
    sample_rate = 16000
    duration = 1.0  # 1 second
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)
    # Generate a simple sine wave with known amplitude
    audio_data = np.sin(2 * np.pi * 440 * t)
    file_path = tmp_path / "test_audio.wav"
    sf.write(file_path, audio_data, sample_rate)
    return file_path


class TestChunkedStridedAudioReader:
    def test_initialization(self, sample_audio_file):
        reader = ChunkedStridedAudioReader(
            audio_file=sample_audio_file,
            chunk_size=4000,
            stride=2000,
            sample_rate=16000,
        )

        assert reader._audio_file.samplerate == 16000
        assert reader.chunk_size == 4000
        assert reader._actual_stride == 2000
        assert reader._current_sample_pos == 0

    def test_num_samples_property(self, sample_audio_file):
        reader = ChunkedStridedAudioReader(
            audio_file=sample_audio_file,
            chunk_size=4000,
            stride=2000,
            sample_rate=16000,
        )

        expected_num_chunks = (16000 - 4000) // 2000
        assert reader.num_chunks == expected_num_chunks

    def test_iterator_protocol(self, sample_audio_file):
        reader = ChunkedStridedAudioReader(
            audio_file=sample_audio_file,
            chunk_size=4000,
            stride=2000,
            sample_rate=16000,
        )

        # Test that class is iterable
        assert iter(reader) is reader

        # Collect all chunks through iteration
        chunks = list(reader)
        assert len(chunks) == reader.num_chunks

        # Verify each chunk
        for chunk in chunks:
            assert isinstance(chunk, torch.Tensor)
            assert chunk.shape[1] == 4000

    def test_resampling(self, sample_audio_file):
        sample_rate = 8000
        reader = ChunkedStridedAudioReader(
            audio_file=sample_audio_file,
            chunk_size=2000,  # Adjust chunk size to match target rate
            stride=1000,
            sample_rate=sample_rate,
        )

        chunk = next(iter(reader))
        assert chunk.shape[1] == 2000

        # Verify overlapping of consecutive chunks
        chunks = list(reader)
        assert len(chunks) > 1

        # Verify chunk size after resampling
        for chunk in chunks:
            assert chunk.shape[1] == 2000

    def test_max_frames_limitation(self, sample_audio_file):
        max_frames = 8000
        reader = ChunkedStridedAudioReader(
            audio_file=sample_audio_file,
            chunk_size=4000,
            stride=2000,
            max_frames=max_frames,
            sample_rate=16000,
        )

        assert reader.num_chunks == (max_frames - 4000) // 2000

    def test_error_handling(self, sample_audio_file):
        # Non positive chunk size
        with pytest.raises(ValueError):
            ChunkedStridedAudioReader(
                audio_file=sample_audio_file,
                chunk_size=0,
                stride=2000,
                sample_rate=16000,
            )

        # Non positive stride
        with pytest.raises(ValueError):
            ChunkedStridedAudioReader(
                audio_file=sample_audio_file,
                chunk_size=4000,
                stride=0,
                sample_rate=16000,
            )

        # Nonexistent file
        with pytest.raises(RuntimeError):
            ChunkedStridedAudioReader(
                audio_file="nonexistent.wav",
                chunk_size=4000,
                stride=2000,
                sample_rate=16000,
            )

    def test_consecutive_reads(self, sample_audio_file):
        reader = ChunkedStridedAudioReader(
            audio_file=sample_audio_file,
            chunk_size=4000,
            stride=2000,
            sample_rate=16000,
        )

        # First chunk
        chunk1 = reader.read()
        assert chunk1 is not None
        assert chunk1.shape[1] == 4000

        # Second chunk should overlap with first
        chunk2 = reader.read()
        assert chunk2 is not None
        assert chunk2.shape[1] == 4000

        # Verify non-None until we reach the end
        chunks = []
        while True:
            chunk = reader.read()
            if chunk is None:
                break
            chunks.append(chunk)
            assert chunk.shape[1] == 4000

        # Verify `None` at end.
        assert reader.read() is None


class TestAudioFilesObservationGenerator:
    @pytest.fixture
    def sample_audio_files(self, tmp_path):
        """Create sample audio files for testing."""
        sample_rate = 16000
        duration = 1.0  # 1 second
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples)

        files = []
        # Create two files with different frequencies
        for i in range(2):
            audio_data = np.sin(2 * np.pi * (440 * (i + 1)) * t)
            file_path = tmp_path / f"test_audio_{i}.wav"
            sf.write(file_path, audio_data, sample_rate)
            files.append(file_path)
        return files

    def test_initialization(self, sample_audio_files):
        generator = AudioFilesObservationGenerator(
            audio_files=sample_audio_files,
            chunk_size=4000,
            stride=2000,
            sample_rate=16000,
            max_frames_per_file=None,
        )
        assert len(generator._audio_readers) == 2
        assert generator._current_reader_index == 0

    def test_num_samples(self, sample_audio_files):
        generator = AudioFilesObservationGenerator(
            audio_files=sample_audio_files,
            chunk_size=4000,
            stride=2000,
            sample_rate=8000,
            max_frames_per_file=None,
        )
        expected_num_chunks = sum((8000 - 4000) // 2000 for _ in range(2))
        assert generator.num_chunks == expected_num_chunks

    def test_max_frames_per_file_list(self, sample_audio_files):
        max_frames = [8000, 12000]
        generator = AudioFilesObservationGenerator(
            audio_files=sample_audio_files,
            chunk_size=4000,
            stride=2000,
            sample_rate=16000,
            max_frames_per_file=max_frames,
        )
        expected_num_chunks = sum((frames - 4000) // 2000 for frames in max_frames)
        assert generator.num_chunks == expected_num_chunks

    def test_max_frames_per_file_single_value(self, sample_audio_files):
        generator = AudioFilesObservationGenerator(
            audio_files=sample_audio_files,
            chunk_size=4000,
            stride=2000,
            sample_rate=16000,
            max_frames_per_file=8000,
        )
        expected_num_chunks = sum((8000 - 4000) // 2000 for _ in range(2))
        assert generator.num_chunks == expected_num_chunks

    def test_invalid_max_frames_per_file(self, sample_audio_files):
        with pytest.raises(ValueError):
            AudioFilesObservationGenerator(
                audio_files=sample_audio_files,
                chunk_size=4000,
                stride=2000,
                sample_rate=16000,
                max_frames_per_file=[8000],  # Wrong length
            )

    def test_sequential_file_processing(self, sample_audio_files):
        generator = AudioFilesObservationGenerator(
            audio_files=sample_audio_files,
            chunk_size=4000,
            stride=2000,
            sample_rate=16000,
            max_frames_per_file=None,
        )

        chunks = []
        try:
            while True:
                chunks.append(generator())
        except StopIteration:
            pass

        assert len(chunks) == generator.num_chunks
        for chunk in chunks:
            assert isinstance(chunk, torch.Tensor)
            assert chunk.shape == (1, 4000)

    def test_empty_file_list(self):
        generator = AudioFilesObservationGenerator(
            audio_files=[],
            chunk_size=4000,
            stride=2000,
            sample_rate=16000,
            max_frames_per_file=None,
        )
        with pytest.raises(StopIteration):
            generator()
