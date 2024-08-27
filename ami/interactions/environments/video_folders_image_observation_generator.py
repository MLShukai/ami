import warnings
from pathlib import Path

import cv2
import torch
from torchvision.transforms import v2


class VideoFoldersImageObservationGenerator:
    """A generator class that provides image observations from multiple video
    folders.

    This class reads video files from specified folders, processes them
    frame by frame, and yields image tensors as observations. It
    supports multiple video formats, custom start frames, and frame
    limits per video folder.
    """

    def __init__(
        self,
        folders: list[str | Path],
        image_size: tuple[int, int],
        extensions: tuple[str, ...] = ("mp4",),
        start_frame_per_videos: int | list[int] = 0,
        video_frames_per_videos: int | None | list[int | None] = None,
    ):
        """Initialize the VideoFoldersImageObservationGenerator.

        Args:
            folders (list[str | Path]): List of folder paths containing video files.
            image_size (tuple[int, int]): Target size for output images (height, width).
            extensions (tuple[str, ...], optional): Tuple of video file extensions to consider. Defaults to ("mp4",).
            start_frame_per_videos (int | list[int], optional): Starting frame(s) for each video folder. Defaults to 0.
            video_frames_per_videos (int | None | list[int | None], optional): Number of frames to use from each video folder.
                                                                               None means use all available frames. Defaults to None.
        """
        self.folders = [Path(folder) for folder in folders]
        self.image_size = image_size
        self.extensions = extensions

        # Validate and set start_frame_per_videos
        if isinstance(start_frame_per_videos, list):
            assert len(start_frame_per_videos) == len(
                self.folders
            ), "Length of start_frame_per_videos must match the number of folders"
            self.start_frame_per_videos = start_frame_per_videos
        else:
            self.start_frame_per_videos = [start_frame_per_videos] * len(self.folders)

        # Validate video_frames_per_videos
        if isinstance(video_frames_per_videos, list):
            assert len(video_frames_per_videos) == len(
                self.folders
            ), "Length of video_frames_per_videos must match the number of folders"
        elif video_frames_per_videos is None:
            video_frames_per_videos = [None] * len(self.folders)
        else:
            video_frames_per_videos = [video_frames_per_videos] * len(self.folders)

        self.video_files = self._get_sorted_video_files()
        self.video_frames_per_videos = self._process_video_frames(video_frames_per_videos)

        self.current_folder_index = 0
        self.current_video_index = 0
        self.current_frame_index = 0
        self.current_video: cv2.VideoCapture | None = None

        self._load_next_video()

    def _get_sorted_video_files(self) -> list[list[Path]]:
        video_files = []
        for folder in self.folders:
            folder_files = []
            for ext in self.extensions:
                folder_files.extend(sorted(folder.glob(f"*.{ext}")))
            video_files.append(folder_files)
        return video_files

    def _process_video_frames(self, video_frames_per_videos: list[int | None]) -> list[int]:
        processed_frames = []
        for folder_index, folder_files in enumerate(self.video_files):
            total_frames = 0
            for file in folder_files:
                cap = cv2.VideoCapture(str(file))
                total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

            start_frame = self.start_frame_per_videos[folder_index]
            available_frames = total_frames - start_frame

            requested_frames = video_frames_per_videos[folder_index]
            if requested_frames is None:
                processed_frames.append(available_frames)
            else:
                assert requested_frames <= available_frames, (
                    f"Folder {self.folders[folder_index]} has fewer available frames ({available_frames}) "
                    f"than requested ({requested_frames})"
                )
                processed_frames.append(requested_frames)

        return processed_frames

    def _load_next_video(self) -> None:
        if self.current_video is not None:
            self.current_video.release()

        # Check if we've processed all folders
        if self.current_folder_index >= len(self.folders):
            raise StopIteration("All videos have been processed")

        # Move to the next folder if we've processed all videos in the current folder
        if self.current_video_index >= len(self.video_files[self.current_folder_index]):
            self.current_folder_index += 1
            self.current_video_index = 0
            return self._load_next_video()

        # Load the next video
        video_path = self.video_files[self.current_folder_index][self.current_video_index]
        self.current_video = cv2.VideoCapture(str(video_path))

        # Set the starting frame
        start_frame = self.start_frame_per_videos[self.current_folder_index]
        self.current_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.current_frame_index = start_frame

    def __call__(self) -> torch.Tensor:
        if self.current_video is None:
            raise StopIteration("All videos have been processed")

        ret, frame = self.current_video.read()
        if not ret:
            warnings.warn(
                f"Failed to read a frame from {self.video_files[self.current_folder_index][self.current_video_index]}, "
                "loading next video file.",
                RuntimeWarning,
            )
            self.current_video_index += 1
            self._load_next_video()
            return self.__call__()

        self.current_frame_index += 1

        # Check if we've reached the frame limit for the current folder
        frames_processed = self.current_frame_index - self.start_frame_per_videos[self.current_folder_index]
        if frames_processed >= self.video_frames_per_videos[self.current_folder_index]:
            self.current_video_index += 1
            self._load_next_video()
            return self.__call__()

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to torch tensor and normalize
        tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0

        # Resize
        tensor = v2.functional.resize(tensor, self.image_size)

        return tensor
