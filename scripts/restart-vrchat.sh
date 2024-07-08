#!/bin/bash

# Check if Steam is running
if ! pgrep -x steam > /dev/null; then
    echo "Alert: Steam is not running. Please start Steam **IN HOST DISPLAY** before running this script."
    exit 1
fi

# Find VRChat process and kill it if found
vrchat_pid=$(pgrep "VRChat")
if [ -n "$vrchat_pid" ]; then
    echo "VRChat process found (PID: $vrchat_pid). Terminating..."
    kill $vrchat_pid
    if [ $? -eq 0 ]; then
        echo "VRChat process successfully terminated."
        echo "Waiting for 10 seconds..."
        sleep 10
    else
        echo "Failed to terminate VRChat process."
    fi
else
    echo "VRChat process not found."
fi

# Launch VRChat through Steam
echo "Launching VRChat..."
steam steam://rungameid/438100
