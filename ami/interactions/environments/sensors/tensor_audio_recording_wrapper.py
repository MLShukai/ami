from typing import Any

from torch import Tensor
from typing_extensions import override

from ...io_wrappers.tensor_audio_recorder import TensorAudioRecorder
from .base_sensor import BaseSensor, BaseSensorWrapper


class TensorAudioRecordingWrapper(BaseSensorWrapper[Tensor, Tensor]):
    """Records the audio tensor data using TensorAudioRecorder."""

    @override
    def __init__(self, sensor: BaseSensor[Tensor], *args: Any, **kwds: Any) -> None:
        """Initializes the TensorAudioRecordingWrapper.

        Args:
            sensor (BaseSensor[Tensor]): Sensor to be wrapped.
            *args (Any): Arguments for TensorAudioRecorder.
            **kwds (Any): Keyword arguments for TensorAudioRecorder.
        """
        super().__init__(sensor)

        self._recoder = TensorAudioRecorder(*args, **kwds)

    @override
    def wrap_observation(self, observation: Tensor) -> Tensor:
        """Writes audio observation to file.

        Args:
            observation (Tensor): Observed audio data. shape is (channels, sample_size)

        Returns:
            Tensor: Input observation
        """
        self._recoder.wrap(observation)
        return observation

    @override
    def setup(self) -> None:
        super().setup()
        self._recoder.setup()

    @override
    def teardown(self) -> None:
        super().teardown()
        self._recoder.teardown()

    @override
    def on_paused(self) -> None:
        super().on_paused()
        self._recoder.on_paused()

    @override
    def on_resumed(self) -> None:
        super().on_resumed()
        self._recoder.on_resumed()
