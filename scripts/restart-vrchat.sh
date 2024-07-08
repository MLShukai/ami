#!/bin/bash

# Find VRChat process and kill it if found
vrchat_pid=$(pgrep -i vrchat)
if [ -n "$vrchat_pid" ]; then
    echo "VRChat process found. Terminating..."
    kill $vrchat_pid
    echo "Waiting for 10 seconds..."
    sleep 10
else
    echo "VRChat process not found."
fi

# Launch VRChat through Steam
echo "Launching VRChat..."
steam steam://rungameid/438100
