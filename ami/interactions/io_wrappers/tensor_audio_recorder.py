import os
from datetime import datetime
from pathlib import Path

import soundfile as sf
from torch import Tensor
from typing_extensions import override

from ami.interactions.environments.sensors.base_sensor import BaseSensor
from ami.logger import get_inference_thread_logger

from .base_io_wrapper import BaseIOWrapper


class TensorAudioRecorder(BaseIOWrapper[Tensor, Tensor]):
    """Records the audio tensor data to wav file."""

    @override
    def __init__(
        self,
        output_dir: str | Path,
        sample_rate: float,
        channel_size: int,
        file_name_format: str = "%Y-%m-%d_%H-%M-%S.%f.wav",
        sample_slice_size: int | None = None,
    ) -> None:
        """Initializes the TensorAudioRecorder.

        Args:
            output_dir (str | Path): Path to the directory where the audio will be saved.
            sample_rate (float, optional): Sampling rate at which the audio is recorded.
            channel_size (int, optional): Channel size at which the audio is recorded.
            file_name_format (str, optional): File name format that includes a datetime format string. Defaults to "%Y-%m-%d_%H-%M-%S.%f.wav".
            sample_slice_size (int | None, optional): Slice size in recording sampled audio observation.. Defaults to None.
        """
        super().__init__()

        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir)
        self.sample_rate = sample_rate
        self.channel_size = channel_size
        self.file_name_format = file_name_format
        self.sample_slice_size = sample_slice_size

        self.logger = get_inference_thread_logger(self.__class__.__name__)

    def open_file_writer(self) -> None:
        self.file_path = self.output_dir / datetime.now().strftime(self.file_name_format)
        self.file_writer = sf.SoundFile(
            self.file_path, mode="w", samplerate=self.sample_rate, channels=self.channel_size, subtype="FLOAT"
        )
        self.logger.info(f"Recording audio to '{self.file_path}'")

    def close_file_writer(self) -> None:
        self.logger.info(f"Saved audio to '{self.file_path}'")
        self.file_writer.close()

    @override
    def wrap(self, input: Tensor) -> Tensor:
        """Writes input audio to file.

        Args:
            input (Tensor): Input audio data. shape is (channels, sample_size)

        Returns:
            Tensor: input.
        """
        record_samples = input.clone().detach_().cpu().transpose_(0, 1)
        record_samples = record_samples.float().numpy()
        if self.sample_slice_size is not None:
            record_samples = record_samples[: self.sample_slice_size]
        self.file_writer.write(record_samples)
        return input

    @override
    def setup(self) -> None:
        super().setup()
        self.open_file_writer()

    @override
    def teardown(self) -> None:
        super().teardown()
        self.close_file_writer()

    @override
    def on_paused(self) -> None:
        super().on_paused()
        self.close_file_writer()  # 経験が途切れるため一度リリース

    @override
    def on_resumed(self) -> None:
        super().on_resumed()
        self.open_file_writer()
