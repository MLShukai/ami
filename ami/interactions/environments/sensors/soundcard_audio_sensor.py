import logging
import math
import os
import threading
import time
from collections import deque

import torch
from torch import Tensor
from typing_extensions import override
from vrchat_io.audio import SoundcardAudioCapture

from .base_sensor import BaseSensor

logger = logging.getLogger(__name__)


class SoundcardAudioSensor(BaseSensor[Tensor]):
    @override
    def __init__(
        self,
        device_name: str | None = None,
        sample_rate: float = 16000,  # 16 kHz
        channel_size: int = 2,
        read_sample_size: int = 1600,  # 0.1 sec
        block_size: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Create SoundcardAudioSensor object for real-time audio capture.

        This sensor captures audio data from a soundcard device in a background thread
        and provides it in a format suitable for deep learning models.

        Args:
            device_name: Soundcard device name. If None, uses the default system device.
            sample_rate: Audio sampling rate in Hz.
            channel_size: Number of audio channels (e.g., 1 for mono, 2 for stereo).
            read_sample_size: Number of samples returned by each `read` call.
                Represents 0.1 seconds of audio at 16kHz by default.
            block_size: Number of samples to read in each background thread iteration.
                Defaults to read_sample_size if None.
            dtype: Torch dtype for the returned audio tensor.

        Raises:
            AssertionError: If read_sample_size, channel_size, sample_rate, or block_size is less than 1.
                block_size is larger than read_sample_size
        """
        super().__init__()
        assert read_sample_size >= 1
        assert channel_size >= 1
        assert sample_rate >= 1
        if block_size is None:
            block_size = read_sample_size
        assert block_size >= 1
        assert block_size <= read_sample_size
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._read_sample_size = read_sample_size
        self._dtype = dtype

        if device_name is None:
            device_name = os.environ.get("AMI_DEFAULT_MICROPHONE")

        if device_name is not None:
            logger.debug(f"Using audio device: {device_name}")

        self._cap = SoundcardAudioCapture(sample_rate, device_name, frame_size=block_size, channels=channel_size)
        self._sampled_blocks: deque[Tensor] = deque(
            [torch.zeros(read_sample_size, channel_size)], maxlen=math.ceil(read_sample_size / block_size)
        )

        self._teardown_flag = threading.Event()
        self._capture_thread = threading.Thread(target=self._read_background)

    def _read_background(self) -> None:
        """Read audio data continuously in a background thread.

        This method runs in a separate thread and continuously captures
        audio data from the soundcard, converting it to torch tensors
        and storing them in the frame blocks deque.

        The thread continues until the teardown flag is set.
        """
        while not self._teardown_flag.is_set():
            samples = self._cap.read()
            self._sampled_blocks.append(torch.from_numpy(samples).type(self._dtype))
            time.sleep(self._block_size / self._sample_rate * 0.1)

    @override
    def setup(self) -> None:
        """Start the background audio capture thread.

        This method must be called before using the sensor to begin
        audio capture.
        """
        super().setup()
        self._capture_thread.start()

    @override
    def read(self) -> Tensor:
        """Read the latest audio frame data.

        Returns:
            Tensor: Audio data tensor of shape (channel_size, read_sample_size).
                The first dimension represents audio channels (e.g., stereo channels),
                and the second dimension represents audio samples in time.
                Values are in the range [-1, 1] for float dtypes.

        Note:
            The returned tensor is transposed from the internal storage format
            to match the expected input format of most audio processing models
            (channels first).

        Raises:
            RuntimeError: If background thread is not started.
        """
        if not self._capture_thread.is_alive():
            raise RuntimeError("Please call method `setup` before `read`!")
        samples = torch.cat(list(self._sampled_blocks))[-self._read_sample_size :]
        return samples.transpose(0, 1)

    @override
    def teardown(self) -> None:
        """Stop the background audio capture and clean up resources.

        This method should be called when the sensor is no longer needed
        to properly close the audio stream and stop the background
        thread.
        """
        super().teardown()
        self._teardown_flag.set()
        self._capture_thread.join()
        del self._cap  # close internal stream.
