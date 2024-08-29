from pathlib import Path
from typing import Iterator

import cv2
import torch


class FolderAsVideo:
    """Handles a folder of video files sequentially as if it were a single
    video file."""

    def __init__(
        self,
        folder: str | Path,
        extensions: tuple[str, ...] = ("mp4",),
        sort_by_name: bool = True,
        start_frame: int = 0,
        max_frames: int | None = None,
    ) -> None:
        """Initialize the FolderAsVideo object.

        Args:
            folder (str | Path): Path to the folder containing video files.
            extensions (tuple[str, ...]): Tuple of video file extensions to consider.
            sort_by_name (bool): Whether to sort video files by name.
            start_frame (int): The frame number to start reading from.
            max_frames (int | None): Maximum number of frames to read. If None, read all available frames.

        Raises:
            ValueError: If start_frame is greater than or equal to total frames,
                        or if max_frames exceeds available frames.
        """
        self.folder = Path(folder)
        self.extensions = extensions
        self.start_frame = start_frame

        self.video_files = self._get_video_files(sort_by_name)
        self.total_frames = self._count_total_frames()

        if self.start_frame >= self.total_frames:
            raise ValueError(f"start_frame ({self.start_frame}) must be less than total frames ({self.total_frames})")

        self.available_frames = self.total_frames - self.start_frame

        if max_frames is None:
            self.max_frames = self.available_frames
        else:
            if max_frames > self.available_frames:
                raise ValueError(f"max_frames ({max_frames}) exceeds available frames ({self.available_frames})")
            self.max_frames = max_frames

        self.current_video_index = 0
        self.current_frame = 0
        self.current_video = None
        self._initialize_video()

    def _get_video_files(self, sort_by_name: bool) -> list[Path]:
        """Get a list of video files in the folder."""
        video_files = []
        for ext in self.extensions:
            video_files.extend(self.folder.glob(f"*.{ext}"))
        return sorted(video_files) if sort_by_name else video_files

    def _count_total_frames(self) -> int:
        """Count the total number of frames across all video files."""
        total = 0
        for video_file in self.video_files:
            cap = self._open_video_file(video_file)
            total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        return total

    @staticmethod
    def _open_video_file(path: Path) -> cv2.VideoCapture:
        """Open a video file and return the VideoCapture object."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file '{path}'")
        return cap

    def _initialize_video(self) -> None:
        """Initialize the video to the correct file and frame position."""
        remaining_frames = self.start_frame
        for i, video_file in enumerate(self.video_files):
            cap = self._open_video_file(video_file)
            frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if remaining_frames < frames_in_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, remaining_frames)
                self.current_video = cap
                self.current_video_index = i
                self.current_frame = self.start_frame
                return
            remaining_frames -= frames_in_video
            cap.release()

        self.current_video = None

    def read(self) -> torch.Tensor | None:
        """Read one frame from the video. Returns None if all frames have been
        read.

        Returns:
            torch.Tensor | None: The read frame as a torch.Tensor, or None if no more frames.

        Raises:
            RuntimeError: If there's an error reading the video file.
        """
        if self.is_finished or self.current_video is None:
            return None

        ret, frame = self.current_video.read()
        if not ret:
            frames_in_current_video = int(self.current_video.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame_in_video = int(self.current_video.get(cv2.CAP_PROP_POS_FRAMES))

            if current_frame_in_video < frames_in_current_video:
                self.current_video.release()
                raise RuntimeError(
                    f"Error reading frame {current_frame_in_video} from video file {self.video_files[self.current_video_index]}"
                )

            self.current_video.release()
            self.current_video_index += 1
            if self.current_video_index < len(self.video_files):
                self.current_video = self._open_video_file(self.video_files[self.current_video_index])
                return self.read()
            else:
                self.current_video = None
                return None

        self.current_frame += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0

    @property
    def is_finished(self) -> bool:
        """Check if all frames have been read."""
        return (self.current_frame - self.start_frame) >= self.max_frames

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame
