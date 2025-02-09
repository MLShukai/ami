import math
from pathlib import Path
from typing import Iterator, TypeAlias

import soundfile as sf
import torch
import torchaudio.functional as F

StrPath: TypeAlias = str | Path


class ChunkedStridedAudioReader:
    """Audio file reader that supports chunked reading with stride.

    This class reads audio files in chunks of specified size with a
    given stride, optionally performing resampling to a target sample
    rate.
    """

    def __init__(
        self,
        audio_file: StrPath,
        chunk_size: int,
        stride: int,
        sample_rate: int,
        max_frames: int | None = None,
    ) -> None:
        """
        Args:
            audio_file: Path to the audio file.
            chunk_size: Number of samples in each chunk.
            stride: Number of samples to advance between chunks.
            sample_rate: Target sample rate for resampling.
            max_frames: Optional; maximum number of frames to read from file (specified sample rate).

        Raises:
            ValueError: If chunk_size or stride is negative.
        """
        if chunk_size < 1:
            raise ValueError
        if stride < 1:
            raise ValueError

        self._audio_file = sf.SoundFile(audio_file)

        if sample_rate < 1:
            raise ValueError

        self._target_sample_rate = sample_rate
        self.chunk_size = chunk_size

        sample_rate_ratio = self._audio_file.samplerate / sample_rate

        self._max_frames = self._audio_file.frames
        if max_frames is not None:
            self._max_frames = min(round(max_frames * sample_rate_ratio), self._max_frames)

        self._actual_chunk_size = math.ceil(chunk_size * sample_rate_ratio)
        if stride is None:
            stride = chunk_size
        self._actual_stride = round(stride * sample_rate_ratio)

        self._current_sample_pos = 0
        self._audio_buffers: list[torch.Tensor] = []

    @property
    def _audio_buffer_length(self) -> int:
        return sum(len(buf) for buf in self._audio_buffers)

    @property
    def num_chunks(self) -> int:
        """Number of valid chunks that can be read from the audio file.

        Returns:
            Total number of chunks considering chunk size and stride.
        """
        return int((self._max_frames - self._actual_chunk_size - 1) / self._actual_stride + 1)

    def read(self) -> torch.Tensor | None:
        """Read one chunk from the audio file.

        Returns:
            A tensor containing the audio chunk of shape (channels, chunk_size),
            or None if all chunks have been read.
        """
        self._current_sample_pos += 1

        if self.num_chunks < self._current_sample_pos:
            return None

        while self._audio_buffer_length < self._actual_chunk_size:
            frame = self._audio_file.read(self._actual_chunk_size, always_2d=True, dtype="float32", fill_value=0.0)
            self._audio_buffers.append(torch.from_numpy(frame))
        buffer = torch.cat(self._audio_buffers)
        out, remain = buffer[: self._actual_chunk_size], buffer[self._actual_stride :]
        self._audio_buffers = [remain]

        return F.resample(out.T, self._audio_file.samplerate, self._target_sample_rate)[:, : self.chunk_size]

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Make the class iterable.

        Returns:
            Iterator over audio chunks.
        """
        return self

    def __next__(self) -> torch.Tensor:
        """Get the next chunk in iteration.

        Returns:
            Next audio chunk as a tensor.

        Raises:
            StopIteration: When all chunks have been read.
        """
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame


class AudioFilesObservationGenerator:
    """Generates audio observations from multiple audio files sequentially.

    This class processes multiple audio files sequentially, yielding
    audio chunks as observations. Each file is processed using
    ChunkedStridedAudioReader with shared chunk size and stride
    parameters.
    """

    def __init__(
        self,
        audio_files: list[StrPath],
        chunk_size: int,
        stride: int,
        sample_rate: int,
        max_frames_per_file: list[int | None] | int | None,
    ) -> None:
        """
        Args:
            audio_files: List of paths to audio files.
            chunk_size: Number of samples in each chunk.
            stride: Number of samples to advance between chunks.
            sample_rate: Target sample rate for resampling.
            max_frames_per_file: Maximum number of frames to read from each file.
                Can be either a single value for all files, a list of values
                per file, or None for no limit.

        Raises:
            ValueError: If max_frames_per_file is a list with length different
                from the number of audio files.
        """
        if isinstance(max_frames_per_file, list):
            if len(max_frames_per_file) != len(audio_files):
                raise ValueError
        else:
            max_frames_per_file = [max_frames_per_file] * len(audio_files)

        self._audio_readers = [
            ChunkedStridedAudioReader(f, chunk_size, stride, sample_rate, max_frames)
            for f, max_frames in zip(audio_files, max_frames_per_file)
        ]
        self._current_reader_index = 0

    @property
    def num_chunks(self) -> int:
        return sum(reader.num_chunks for reader in self._audio_readers)

    def __call__(self) -> torch.Tensor:
        # if self._current_reader_index >= len(self._audio_readers):
        #     raise StopIteration
        self._current_reader_index = self._current_reader_index % len(self._audio_readers)
        reader = self._audio_readers[self._current_reader_index]
        sample = reader.read()
        if sample is None:
            self._current_reader_index += 1
            return self.__call__()
        else:
            return sample
