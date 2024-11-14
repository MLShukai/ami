#!/bin/bash

# This script automatically restarts VRChat and sets up OBS at specified intervals.
# It requires restart-vrchat.sh and obs_vrchat_setup.py to be in the same directory.

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if required commands exist
if ! command_exists curl || ! command_exists pgrep || ! command_exists kill || ! command_exists steam || ! command_exists python3; then
    echo "Error: Required commands (curl, pgrep, kill, steam, or python3) not found. Please install them."
    exit 1
fi

# Check if the Python script exists
if [ ! -f "${SCRIPT_DIR}/obs_vrchat_setup.py" ]; then
    echo "Error: obs_vrchat_setup.py not found in the script directory."
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/auto_control_vrchat_menu.py" ]; then
    echo "Error: auto_control_vrchat_menu.py not found in the script directory."
    exit 1
fi

# Check if the restart script exists
if [ ! -f "${SCRIPT_DIR}/restart-vrchat.sh" ]; then
    echo "Error: restart-vrchat.sh not found in the script directory."
    exit 1
fi

# Function to restart VRChat and setup OBS
restart_and_setup() {
    echo "Restarting VRChat and setting up OBS..."

    # Pause the AMI
    curl -X POST http://localhost:8391/api/pause

    # Run the restart-vrchat.sh script
    bash "${SCRIPT_DIR}/restart-vrchat.sh"

    # Wait for VRChat to start (adjust this time if needed)
    echo "Waiting for VRChat to start..."
    sleep 60

    # Run the Python script to setup OBS
    python3 "${SCRIPT_DIR}/obs_vrchat_setup.py"

    # Run the Python script to control VRChat menu
    python3 "${SCRIPT_DIR}/auto_control_vrchat_menu.py" --scenario "${SCRIPT_DIR}/vrchat_images/create_and_join_group_public_japan_street"

    # Resume the AMI
    curl -X POST http://localhost:8391/api/resume

    echo "Cycle completed."
}

# Main loop
while true; do
    # Ask user for the cycle time in minutes
    read -p "Enter the cycle time in minutes (or 'q' to quit): " cycle_time

    # Check if user wants to quit
    if [ "$cycle_time" = "q" ]; then
        echo "Exiting script."
        exit 0
    fi

    # Validate input
    if ! [[ "$cycle_time" =~ ^[0-9]+$ ]]; then
        echo "Invalid input. Please enter a positive integer."
        continue
    fi

    # Convert minutes to seconds
    seconds=$((cycle_time * 60))

    echo "Starting cycle. Will restart VRChat and setup OBS every $cycle_time minutes."

    # Initial run
    restart_and_setup

    # Loop until interrupted
    while true; do
        echo "Sleeping for $cycle_time minutes..."
        sleep $seconds
        restart_and_setup
    done
done
