#!/bin/bash

# Check if VRChat is running and kill it if found
if pgrep VRChat > /dev/null; then
    killall VRChat
    echo "VRChat has been terminated. Waiting for 10 seconds..."
    sleep 10
else
    echo "VRChat is not running."
fi

# Launch VRChat through Steam
echo "Launching VRChat..."
steam steam://rungameid/438100