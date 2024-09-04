from pathlib import Path
from typing import Iterator

import cv2
import torch
from torchvision.transforms import v2


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
        normalize: bool = True,
    ) -> None:
        """Initialize the FolderAsVideo object.

        Args:
            folder (str | Path): Path to the folder containing video files.
            extensions (tuple[str, ...]): Tuple of video file extensions to consider.
            sort_by_name (bool): Whether to sort video files by name.
            start_frame (int): The frame number to start reading from.
            max_frames (int | None): Maximum number of frames to read. If None, read all available frames.
            normalize (bool): Whether to normalize the value range to [0,1].

        Raises:
            ValueError: If start_frame is greater than or equal to total frames,
                        or if max_frames exceeds available frames.
        """
        self.folder = Path(folder)
        self.extensions = extensions
        self.start_frame = start_frame
        self.normalize = normalize

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
        self.current_video: cv2.VideoCapture
        self._initialize_video()

    def _get_video_files(self, sort_by_name: bool) -> list[Path]:
        """Get a list of video files in the folder."""
        video_files: list[Path] = []
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

    def read(self) -> torch.Tensor:
        """Read one frame from the video.

        Returns:
            torch.Tensor: The read frame as a torch.Tensor.

        Raises:
            RuntimeError: If there's an error reading the video file or if all frames have been read.
        """
        if self.is_finished:
            raise RuntimeError("All frames have been read")

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
                raise RuntimeError("All frames have been read")

        self.current_frame += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
        if self.normalize:
            frame_tensor = frame_tensor.float() / 255.0
        return frame_tensor

    @property
    def is_finished(self) -> bool:
        """Check if all frames have been read."""
        return (self.current_frame - self.start_frame) >= self.max_frames

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        if self.is_finished:
            raise StopIteration
        return self.read()


class VideoFoldersImageObservationGenerator:
    """Generates image observations from multiple video folders.

    This class processes video files from specified folders
    sequentially, yielding image tensors as observations. It supports
    various video formats, custom starting frames, and frame limits for
    each video folder.
    """

    def __init__(
        self,
        folder_paths: list[str | Path],
        image_size: tuple[int, int],
        video_extensions: tuple[str, ...] = ("mp4",),
        folder_start_frames: int | list[int] = 0,
        folder_frame_limits: int | None | list[int | None] = None,
        normalize: bool = True,
    ):
        """Initialize the VideoFoldersImageObservationGenerator.

        Args:
            folder_paths: List of paths to folders containing video files.
            image_size: Desired size for output images (height, width).
            video_extensions: Tuple of video file extensions to process.
            folder_start_frames: Starting frame(s) for each video folder.
            folder_frame_limits: Maximum number of frames to use from each folder.
                                 None means use all available frames.
            normalize (bool): Whether to normalize the value range to [0,1].

        Raises:
            ValueError: If the length of folder_start_frames or folder_frame_limits
                        doesn't match the number of folders when provided as lists.
        """
        self.folder_paths = [Path(folder) for folder in folder_paths]
        self.image_size = image_size

        # Process start frames
        if isinstance(folder_start_frames, int):
            self.folder_start_frames = [folder_start_frames] * len(self.folder_paths)
        else:
            if len(folder_start_frames) != len(self.folder_paths):
                raise ValueError("Length of folder_start_frames must match the number of folders")
            self.folder_start_frames = folder_start_frames

        # Process frame limits
        if folder_frame_limits is None or isinstance(folder_frame_limits, int):
            self.folder_frame_limits = [folder_frame_limits] * len(self.folder_paths)
        else:
            if len(folder_frame_limits) != len(self.folder_paths):
                raise ValueError("Length of folder_frame_limits must match the number of folders")
            self.folder_frame_limits = folder_frame_limits

        # Initialize FolderAsVideo objects for each folder
        self.folder_videos = [
            FolderAsVideo(folder, video_extensions, True, start_frame, max_frames, normalize)
            for folder, start_frame, max_frames in zip(
                self.folder_paths, self.folder_start_frames, self.folder_frame_limits
            )
        ]

        self.current_folder_index = 0

    @property
    def max_frames(self) -> int:
        return sum(f.max_frames for f in self.folder_videos)

    def __call__(self) -> torch.Tensor:
        """Generate the next image observation.

        Returns:
            A tensor representing the next video frame.

        Raises:
            StopIteration: When all videos in all folders have been processed.
        """
        while self.current_folder_index < len(self.folder_videos):
            current_video = self.folder_videos[self.current_folder_index]
            if not current_video.is_finished:
                frame = current_video.read()
                # Resize the frame to the target size
                return v2.functional.resize(frame, self.image_size)
            else:
                # Move to the next folder
                self.current_folder_index += 1

        raise StopIteration("All videos have been processed")

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        return self.__call__()
