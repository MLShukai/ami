import cv2
import numpy as np
import pytest
import torch

from ami.interactions.environments.video_folders_image_observation_generator import (
    VideoFoldersImageObservationGenerator,
)

MAX_FRAMES = 60


@pytest.fixture
def sample_video_folders(tmp_path):
    folders = [tmp_path / "folder1", tmp_path / "folder2"]
    for folder in folders:
        folder.mkdir()
        for i in range(2):
            video_path = folder / f"video_{i}.mp4"
            # ダミーのビデオファイルを作成
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (100, 100))
            for _ in range(MAX_FRAMES):  # 各ビデオに60フレーム
                frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
    return folders


class TestVideoFoldersImageObservationGenerator:
    def test_image_size(self, sample_video_folders):
        image_size = (32, 48)
        generator = VideoFoldersImageObservationGenerator(sample_video_folders, image_size)
        result = generator()
        assert result.shape == (3, *image_size)

    def test_output_tensor_properties(self, sample_video_folders):
        generator = VideoFoldersImageObservationGenerator(sample_video_folders, (64, 64))
        result = generator()
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float
        assert torch.all(result >= 0) and torch.all(result <= 1)

    @pytest.mark.parametrize("start_frame", [0, 10, 30])
    def test_start_frame(self, sample_video_folders, start_frame):
        generator = VideoFoldersImageObservationGenerator(
            sample_video_folders, (64, 64), start_frame_per_videos=start_frame
        )
        result = generator()
        assert isinstance(result, torch.Tensor)

    @pytest.mark.parametrize("frames_to_use", [None, 20, 50])
    def test_frames_to_use(self, sample_video_folders, frames_to_use):
        generator = VideoFoldersImageObservationGenerator(
            sample_video_folders, (64, 64), video_frames_per_videos=frames_to_use
        )
        results = [generator() for _ in range(frames_to_use or MAX_FRAMES)]  # Noneの場合は60フレーム
        assert len(results) == (frames_to_use or MAX_FRAMES)

    def test_multiple_folders_with_different_configs(self, sample_video_folders):
        generator = VideoFoldersImageObservationGenerator(
            sample_video_folders, (64, 64), start_frame_per_videos=[0, 10], video_frames_per_videos=[30, None]
        )
        results = [generator() for _ in range(80)]  # 30 from first folder, 50 (60 - 10) from second
        assert len(results) == 80
        assert all(isinstance(r, torch.Tensor) for r in results)

    def test_stopiteration(self, sample_video_folders):
        generator = VideoFoldersImageObservationGenerator(sample_video_folders, (64, 64), video_frames_per_videos=10)
        with pytest.raises(StopIteration):
            for _ in range(50):  # 2 folders * 2 videos * 10 frames = 40, so this should raise StopIteration
                generator()
