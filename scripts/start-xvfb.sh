#!/bin/bash
Xvfb :99 -ac -screen 0 1280x720x24 &
export DISPLAY=:99
exec "$@"
