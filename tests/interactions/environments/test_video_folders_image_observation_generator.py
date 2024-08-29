import cv2
import numpy as np
import pytest
import torch

from ami.interactions.environments.video_folders_image_observation_generator import (
    FolderAsVideo,
    VideoFoldersImageObservationGenerator,
)


@pytest.fixture
def video_folder(tmp_path):
    folder = tmp_path / "test_videos"
    folder.mkdir()
    create_test_video(folder / "video1.mp4", num_frames=10)
    create_test_video(folder / "video2.mp4", num_frames=10)
    return folder


def create_test_video(file_path, num_frames=10, width=64, height=64):
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, 30, (width, height))

    for i in range(num_frames):
        frame = np.full((height, width, 3), 255, dtype=np.uint8)  # Each frame has a different color
        out.write(frame)

    out.release()


class TestFolderAsVideo:
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


class TestVideoFoldersImageObservationGenerator:
    def test_basic_functionality(self, video_folder):
        generator = VideoFoldersImageObservationGenerator(folder_paths=[video_folder], image_size=(32, 32))

        frames = []
        for _ in range(20):
            frames.append(generator())

        assert len(frames) == 20
        for frame in frames:
            assert isinstance(frame, torch.Tensor)
            assert frame.shape == (3, 32, 32)

    def test_multiple_folders(self, tmp_path):
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        create_test_video(folder1 / "video1.mp4", num_frames=5)
        create_test_video(folder2 / "video2.mp4", num_frames=5)

        generator = VideoFoldersImageObservationGenerator(folder_paths=[folder1, folder2], image_size=(32, 32))

        frames = []
        for _ in range(10):
            frames.append(generator())

        assert len(frames) == 10

    def test_start_frames(self, video_folder):
        generator = VideoFoldersImageObservationGenerator(
            folder_paths=[video_folder], image_size=(32, 32), folder_start_frames=5
        )

        frames = list(generator)
        assert len(frames) == 15

    def test_frame_limits(self, video_folder):
        generator = VideoFoldersImageObservationGenerator(
            folder_paths=[video_folder], image_size=(32, 32), folder_frame_limits=15
        )

        frames = list(generator)
        assert len(frames) == 15

    def test_start_frames_and_frame_limits(self, video_folder):
        generator = VideoFoldersImageObservationGenerator(
            folder_paths=[video_folder], image_size=(32, 32), folder_start_frames=5, folder_frame_limits=10
        )

        frames = list(generator)
        assert len(frames) == 10

    def test_different_settings_for_multiple_folders(self, tmp_path):
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        create_test_video(folder1 / "video1.mp4", num_frames=10)
        create_test_video(folder2 / "video2.mp4", num_frames=10)

        generator = VideoFoldersImageObservationGenerator(
            folder_paths=[folder1, folder2],
            image_size=(32, 32),
            folder_start_frames=[0, 5],
            folder_frame_limits=[5, None],
        )

        frames = list(iter(generator))
        assert len(frames) == 10  # 5 from first folder, 5 from second folder

    def test_stop_iteration(self, video_folder):
        generator = VideoFoldersImageObservationGenerator(folder_paths=[video_folder], image_size=(32, 32))

        for _ in range(20):
            generator()

        with pytest.raises(StopIteration):
            generator()

    def test_image_resizing(self, video_folder):
        generator = VideoFoldersImageObservationGenerator(folder_paths=[video_folder], image_size=(128, 128))

        frame = generator()
        assert frame.shape == (3, 128, 128)

    def test_invalid_input(self, video_folder):
        with pytest.raises(ValueError):
            VideoFoldersImageObservationGenerator(
                folder_paths=[video_folder],
                image_size=(32, 32),
                folder_start_frames=[0, 5],  # This should raise an error as there's only one folder
            )

        with pytest.raises(ValueError):
            VideoFoldersImageObservationGenerator(
                folder_paths=[video_folder],
                image_size=(32, 32),
                folder_frame_limits=[10, 20],  # This should raise an error as there's only one folder
            )

    def test_frame_values(self, video_folder):
        generator = VideoFoldersImageObservationGenerator(folder_paths=[video_folder], image_size=(64, 64))

        for _ in range(20):
            frame = generator()
            assert torch.allclose(frame, torch.full_like(frame, 1.0), atol=1e-1)
