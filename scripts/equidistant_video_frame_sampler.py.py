"""Video Frame Sampler.

This script samples frames from video folders and saves them as JPEG images.

Usage:
    python script_name.py --folder_paths PATH [PATH ...] --output_dir PATH
                          [--image_size WIDTH HEIGHT] [--folder_frame_limits LIMIT]
                          [--num_sample SAMPLES]

Arguments:
    --folder_paths PATH [PATH ...]
        Paths to one or more folders containing video files. Required.

    --output_dir PATH
        Directory where sampled frames will be saved as JPEG images. Required.

    --image_size WIDTH HEIGHT
        Size of the output images in pixels. Default is 256 256.

    --folder_frame_limits LIMIT
        Maximum number of frames to process from each folder. Default is 14400 (4 hours at 60 fps).

    --num_sample SAMPLES
        Number of frames to sample across all videos. Default is 65536 (2^16).

Example:
    python video_frame_sampler.py --folder_paths /path/to/videos1 /path/to/videos2
                                  --output_dir /path/to/output
                                  --image_size 512 512
                                  --folder_frame_limits 7200
                                  --num_sample 10000

Note:
    The script uses the VideoFoldersImageObservationGenerator from the ami.interactions.environments
    module to process video frames. Ensure you have the necessary dependencies installed.
"""

import argparse
from pathlib import Path

from torchvision.io import write_jpeg

from ami.interactions.environments.video_folders_image_observation_generator import (
    VideoFoldersImageObservationGenerator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample frames from video folders")
    parser.add_argument("--folder-paths", nargs="+", required=True, help="Paths to video folders")
    parser.add_argument("--image-size", nargs=2, type=int, default=[144, 144], help="Output image size (width height)")
    parser.add_argument("--folder-frame-limits", type=int, default=60 * 60 * 4 * 10, help="Frame limit per folder")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for sampled frames")
    parser.add_argument("--num-sample", type=int, default=2**16, help="Number of frames to sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generator = VideoFoldersImageObservationGenerator(
        folder_paths=args.folder_paths,
        image_size=tuple(args.image_size),
        folder_start_frames=0,
        folder_frame_limits=args.folder_frame_limits,
        normalize=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Available frames: ", generator.max_frames)
    frame_write_interval = generator.max_frames // args.num_sample
    print("Frame interval: ", frame_write_interval)

    sample_count = 0
    for i, frame in enumerate(generator):
        if i % frame_write_interval == 0:
            sample_count += 1
            name = str(i).zfill(len(str(generator.max_frames)))
            write_jpeg(frame, str(args.output_dir / f"{name}.jpeg"))
            print(f"\r{sample_count / args.num_sample * 100:.2f}%", end="", flush=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
