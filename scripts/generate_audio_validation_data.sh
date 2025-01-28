#!/bin/bash

# Must run the host OS.
LOG_DIR="/mnt/Archive/ami-logs/random_observation_action_log.20241123"
SCRIPT_DIR=$(dirname $0)

python $SCRIPT_DIR/equidistant_audio_chunk_sampler.py \
    --chunk-size 32000 \
    --stride 1600 \
    --sample-rate 16000 \
    --num-chunks 65536 \
    --audio-paths $LOG_DIR/**/io/audio_recordings/*.wav \
    --output-dir $LOG_DIR/validation
