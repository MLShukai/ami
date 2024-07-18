#!/bin/bash

# Usage
if [ $# -lt 2 ]; then
    echo "Usage: $0 <scale> <command> [args...]"
    exit 1
fi

# Time scale and command
TIME_SCALE=$1
shift
COMMAND="$@"

# Execute command with libfaketime
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1 FAKETIME="x${TIME_SCALE}" $COMMAND
