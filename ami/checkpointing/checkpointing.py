from datetime import datetime
from pathlib import Path
from typing import TypeAlias

from ..logger import get_main_thread_logger
from ..threads.base_thread import BaseThread

StrPath: TypeAlias = str | Path


class Checkpointing:
    """Handles the saving and loading of checkpoints.

    This class operates on a thread object, utilizing its `save_state`
    and `load_state` methods to manage checkpoints.
    """

    def __init__(self, checkpoints_dir: StrPath, checkpoint_name_format: str = "%Y-%m-%d_%H-%M-%S,%f.ckpt") -> None:
        """Initializes the Checkpointing class.

        Args:
            checkpoints_dir: The directory to store checkpoints.
            checkpoint_name_format: The datetime format used for naming checkpoint files.
        """

        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.checkpoint_name_format = checkpoint_name_format
        self._threads: list[BaseThread] = []

        self._logger = get_main_thread_logger(self.__class__.__name__)

    def add_threads(self, *threads: BaseThread) -> None:
        """Adds thread objects whose state will be saved and loaded."""
        self._threads.extend(threads)

    def save_checkpoint(self) -> Path:
        """Saves the current state of all threads in a new checkpoint named
        with the current time.

        Returns:
            Path: The path to the saved checkpoint directory.
        """
        checkpoint_path = self.checkpoints_dir / datetime.now().strftime(self.checkpoint_name_format)
        if checkpoint_path.exists():
            self._logger.warning(
                f"Aborted saving checkpoint because the checkpoint path already exists.: '{checkpoint_path}.'"
            )
            return checkpoint_path

        checkpoint_path.mkdir()
        for thread in self._threads:
            thread.save_state(checkpoint_path / thread.thread_name)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: StrPath) -> None:
        """Loads a checkpoint from the specified path.

        Args:
            checkpoint_path: The directory path of the checkpoint to load.

        Raises:
            FileNotFoundError: If the specified checkpoint path is not found.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path: '{checkpoint_path}' not found!")

        for thread in self._threads:
            thread.load_state(checkpoint_path / thread.thread_name)
