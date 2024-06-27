from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from queue import Queue
from threading import Thread

import h5py
import torch
from torch import Tensor
from typing_extensions import override

from ami.logger import get_inference_thread_logger

from .base_actuator import BaseActuator, BaseActuatorWrapper


class ControlCommands(Enum):
    """Enumeration for control commands."""

    TEARDOWN = auto()
    FLUSH = auto()


class TensorActionHDF5Recorder(BaseActuatorWrapper[Tensor, Tensor]):
    """Records tensor actions to an HDF5 file.

    This class wraps a BaseActuator and records its actions (which are
    tensors) into an HDF5 file. Actions are queued and written to the
    file in batches to optimize performance and reduce I/O operations.
    """

    @override
    def __init__(
        self,
        actuator: BaseActuator[Tensor],
        file_path: Path,
        flush_batch_size: int = 100,
        batch_name_format: str = "written_time.%Y-%m-%d_%H-%M-%S.%f",
        recording_dtype: torch.dtype | None = None,
        recording_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Initializes the TensorActionHDF5Recorder.

        Args:
            actuator: The base actuator to wrap and record actions from.
            file_path: Path to the HDF5 file where actions will be recorded.
            flush_batch_size: Number of actions to accumulate before flushing to the file.
            batch_name_format: Format string for naming batches in the HDF5 file.
            recording_dtype: Data type to convert tensors before recording.
            recording_shape: Shape to reshape tensors before recording.
        """
        super().__init__(actuator)
        self._logger = get_inference_thread_logger(self.__class__.__name__)
        self._file_path = Path(file_path)
        self._flush_batch_size = flush_batch_size
        self._batch_name_format = batch_name_format
        self._recording_dtype = recording_dtype
        self._recording_shape = recording_shape

        self._current_batch: list[Tensor] = []
        self._data_or_command_queue: Queue[Tensor | ControlCommands] = Queue()

    def writer(self) -> None:
        """Writer thread function.

        This function runs in a separate thread, continuously processing
        the queue to write tensors to the HDF5 file. It handles control
        commands to flush the current batch or terminate the thread.
        """
        running = True
        self._logger.info(f"Saving actions to '{self._file_path}'")
        with h5py.File(self._file_path, "a") as file_writer:
            while running:
                match value := self._data_or_command_queue.get():
                    case ControlCommands.TEARDOWN:
                        running = False
                    case ControlCommands.FLUSH:
                        self.flush_batch(file_writer)
                    case Tensor():
                        self._current_batch.append(value)
                        if len(self._current_batch) % self._flush_batch_size == 0:
                            self.flush_batch(file_writer)

            self.flush_batch(file_writer)

    def flush_batch(self, file_writer: h5py.File) -> None:
        """Flushes the current batch of tensors to the HDF5 file.

        Args:
            file_writer: The HDF5 file writer object.
        """
        if len(self._current_batch) == 0:
            return
        self._logger.info(f"Flushing batch, data size: {len(self._current_batch)}")

        batch = torch.stack(self._current_batch).numpy()
        batch_name = datetime.now().strftime(self._batch_name_format)
        file_writer.create_dataset(batch_name, data=batch)
        self._current_batch.clear()

    @override
    def wrap_action(self, action: Tensor) -> Tensor:
        """Wraps the given action tensor.

        This function clones the action tensor, optionally changes its data type and shape, and puts it into the queue.

        Args:
            action: The action tensor to be recorded.

        Returns:
            The original action tensor.
        """
        put_action = action.cpu().clone()
        if self._recording_dtype is not None:
            put_action = put_action.type(self._recording_dtype)
        if self._recording_shape is not None:
            put_action = put_action.view(self._recording_shape)
        self._data_or_command_queue.put(put_action)
        return action

    @override
    def setup(self) -> None:
        """Sets up the recorder by starting the writer thread."""
        super().setup()
        self._writer_thread = Thread(target=self.writer)
        self._writer_thread.start()

    @override
    def teardown(self) -> None:
        """Tears down the recorder by stopping the writer thread and closing
        the HDF5 file."""
        super().teardown()
        self._data_or_command_queue.put(ControlCommands.TEARDOWN)
        self._writer_thread.join()

    @override
    def on_paused(self) -> None:
        """Handles the pause event by flushing the current batch to the HDF5
        file."""
        super().on_paused()
        self._data_or_command_queue.put(ControlCommands.FLUSH)
