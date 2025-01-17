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
        target_sample_rate: int | None = None,
        max_frames: int | None = None,
    ) -> None:
        """
        Args:
            audio_file: Path to the audio file.
            chunk_size: Number of samples in each chunk.
            stride: Number of samples to advance between chunks.
            target_sample_rate: Optional; target sample rate for resampling.
            max_frames: Optional; maximum number of frames to read from file (specified sample rate).

        Raises:
            ValueError: If chunk_size or stride is negative.
        """
        if chunk_size < 1:
            raise ValueError
        if stride < 1:
            raise ValueError

        self._audio_file = sf.SoundFile(audio_file)
        if target_sample_rate is None:
            target_sample_rate = self._audio_file.samplerate

        if target_sample_rate < 1:
            raise ValueError

        self._target_sample_rate = target_sample_rate
        self.chunk_size = chunk_size

        self._max_frames = self._audio_file.frames
        if max_frames is not None:
            self._max_frames = max_frames

        sample_rate_ratio = self._audio_file.samplerate / target_sample_rate
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
    def num_samples(self) -> int:
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

        if self.num_samples < self._current_sample_pos:
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
