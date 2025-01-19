"""Audio Frame Sampler.

This script samples audio chunks from audio files and saves them as WAV files.

Usage:
    python equidistant_audio_sampler.py --audio_paths PATH [PATH ...] --output_dir PATH
                                       [--chunk_size SAMPLES] [--stride SAMPLES]
                                       [--target_rate RATE] [--max_frames_per_file LIMIT]
                                       [--num_chunks SAMPLES]

Arguments:
    --audio_paths PATH [PATH ...]
        Paths to one or more audio files. Required.

    --output_dir PATH
        Directory where sampled chunks will be saved as WAV files. Required.

    --chunk_size SAMPLES
        Number of samples in each chunk. Default is 16000 (1 second at 16kHz).

    --stride SAMPLES
        Number of samples to advance between chunks. Default is equal to chunk_size.

    --target_rate RATE
        Target sample rate for resampling. Default is 16kHz.

    --max_frames_per_file LIMIT
        Maximum number of frames to process from each file. Default is None.

    --num_chunks SAMPLES
        Number of chunks to sample across all audio files. Default is 65536 (2^16).
"""

import argparse
from pathlib import Path

import torchaudio

from ami.interactions.environments.audio_observation_generator import (
    AudioFilesObservationGenerator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample chunks from audio files")
    parser.add_argument(
        "--audio-paths",
        nargs="+",
        required=True,
        help="Paths to audio files",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16000,
        help="Number of samples in each chunk",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Number of samples to advance between chunks",
    )
    parser.add_argument(
        "--target-rate",
        type=int,
        default=16000,
        help="Target sample rate for resampling",
    )
    parser.add_argument(
        "--max-frames-per-file",
        type=int,
        default=None,
        help="Frame limit per file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for sampled chunks",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=2**16,
        help="Number of chunks to sample",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stride = args.stride if args.stride is not None else args.chunk_size

    generator = AudioFilesObservationGenerator(
        audio_files=args.audio_paths,
        chunk_size=args.chunk_size,
        stride=stride,
        target_sample_rate=args.target_rate,
        max_frames_per_file=args.max_frames_per_file,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Available chunks: ", generator.num_chunks)
    chunk_write_interval = max(1, generator.num_chunks // args.num_chunks)
    print("Chunk interval: ", chunk_write_interval)

    sample_count = 0

    try:
        chunk_index = 0
        while True:
            chunk = generator()
            if chunk_index % chunk_write_interval == 0:
                sample_count += 1
                name = str(chunk_index).zfill(len(str(generator.num_chunks)))
                torchaudio.save(
                    args.output_dir / f"{name}.wav",
                    chunk,
                    args.target_rate,
                )
                print(f"\r{sample_count / min(args.num_chunks, generator.num_chunks) * 100:.2f}%", end="", flush=True)
            chunk_index += 1

    except StopIteration:
        pass

    print("\nDone!")


if __name__ == "__main__":
    main()
