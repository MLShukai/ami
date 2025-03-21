import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset


class IntervalSamplingAudioDataset(Dataset[tuple[torch.Tensor]]):
    """Dataset class for sampling audios at regular intervals from a directory.

    This class creates a dataset by sampling audios at fixed intervals from a specified directory.
    It supports .wav formats and can optionally pre-load audios into memory for faster access.

    Key Features:
    - Samples audios at regular intervals from a directory.
    - Supports .wav format.
    - Applies specified transformations to the audios.
    - Option to pre-load audios into memory for faster access.

    Usage:
    - Initialize the dataset with the desired parameters.
    - Use it with PyTorch DataLoader for efficient batch processing.

    Example:
        dataset = IntervalSamplingAudioDataset("/path/to/audios", sample_rate=16000, num_select=1000)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    Note:
    - Pre-loading audios can significantly increase memory usage for large datasets.
    """

    def __init__(
        self,
        audio_dir: str | Path,
        sample_rate: int,
        sample_size: int,
        num_select: int,
        pre_loading: bool = True,
    ) -> None:
        """Initializes the IntervalSamplingAudioDataset.

        Args:
            audio_dir: Directory containing the audios.
            sample_rate: Sample rate of output audio.
            sample_size: Sample size of output audio.
            num_select: Target number of .wav files to select.
            pre_loading: If True, pre-loads all audios into memory.
        """
        super().__init__()
        self.audio_dir = Path(audio_dir)
        assert self.audio_dir.is_dir()
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.num_select = num_select
        self.audio_files = self._sample_audio_files()
        self.audio_data: list[torch.Tensor] | None = None
        if pre_loading:
            self.audio_data = [self._read_audio(f) for f in self.audio_files]

    def _list_audio_files(self) -> list[Path]:
        files: list[Path] = []
        files.extend(self.audio_dir.glob("*.wav"))
        return sorted(files)

    def _sample_audio_files(self) -> list[Path]:
        available_audio_files = self._list_audio_files()
        if len(available_audio_files) < self.num_select:
            warnings.warn(
                f"num_select({self.num_select}) is larger than num of available audio files({len(available_audio_files)})."
            )
        interval = max(len(available_audio_files) // self.num_select, 1)

        files: list[Path] = []
        for i, file in enumerate(available_audio_files):
            if len(files) == self.num_select:
                break
            if i % interval == 0:
                files.append(file)
        return files

    def _read_audio(self, file: Path) -> torch.Tensor:
        audio, _sample_rate = torchaudio.load(str(file))
        if _sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, _sample_rate, self.sample_rate)
        # Make length into self.sample_size.
        if audio.size(-1) < self.sample_size:
            audio = nn.functional.pad(audio, pad=[0, self.sample_size - audio.size(-1)], mode="constant", value=0.0)
        audio = audio[..., : self.sample_size]
        return audio

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        if self.audio_data is None:
            audio = self._read_audio(self.audio_files[index])
        else:
            audio = self.audio_data[index]
        return (audio,)  # for adjusting batch format to `TensorDataset`.
