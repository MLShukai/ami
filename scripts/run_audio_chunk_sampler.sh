#!/bin/bash

SCRIPT_DIR=$(dirname $0)

python $SCRIPT_DIR/equidistant_audio_chunk_sampler.py \
    --chunk-size 32000 \
    --stride 1600 \
    --target-rate 16000 \
    --num-sample 65536 \
    --audio-paths /mnt/Archive/ami-logs/random_observation_action_log.20241123/**/io/audio_recordings/*.wav \
    --output-dir /mnt/Archive/ami-logs/random_observation_action_log.20241123/validation
