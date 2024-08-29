import cv2
import numpy as np
import pytest
import torch

from ami.interactions.environments.video_folders_image_observation_generator import (
    FolderAsVideo,
)


class TestFolderAsVideo:
    @pytest.fixture
    def video_folder(self, tmp_path):
        folder = tmp_path / "test_videos"
        folder.mkdir()
        self.create_test_video(folder / "video1.mp4", num_frames=10)
        self.create_test_video(folder / "video2.mp4", num_frames=10)
        return folder

    @staticmethod
    def create_test_video(file_path, num_frames=10, width=64, height=64):
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(str(file_path), fourcc, 30, (width, height))

        for _ in range(num_frames):
            frame = np.full((height, width, 3), 255, dtype=np.uint8)  # Each frame has a different color
            out.write(frame)

        out.release()

    def test_basic_functionality(self, video_folder):
        folder_video = FolderAsVideo(video_folder)
        assert folder_video.total_frames == 20

        frames = list(folder_video)
        assert len(frames) == 20

        # Check if frames are correctly read
        for i, frame in enumerate(frames):
            assert isinstance(frame, torch.Tensor)
            assert frame.shape == (3, 64, 64)
            assert torch.allclose(frame, torch.tensor(1.0), atol=1e-1)

    def test_start_frame(self, video_folder):
        folder_video = FolderAsVideo(video_folder, start_frame=5)
        frames = list(folder_video)
        assert len(frames) == 15

    def test_max_frames(self, video_folder):
        folder_video = FolderAsVideo(video_folder, max_frames=15)
        frames = list(folder_video)
        assert len(frames) == 15

    def test_start_frame_and_max_frames(self, video_folder):
        folder_video = FolderAsVideo(video_folder, start_frame=5, max_frames=10)
        frames = list(folder_video)
        assert len(frames) == 10

    def test_init_errors(self, video_folder):

        with pytest.raises(ValueError):
            FolderAsVideo(video_folder, start_frame=999999)

        with pytest.raises(ValueError):
            FolderAsVideo(video_folder, max_frames=999999)

    def test_overread_error(self, video_folder):
        folder_video = FolderAsVideo(video_folder, start_frame=0, max_frames=5)
        for _ in range(5):
            folder_video.read()

        assert folder_video.is_finished
        with pytest.raises(RuntimeError):
            folder_video.read()
