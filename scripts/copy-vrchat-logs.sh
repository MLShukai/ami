#!/bin/bash

# VRChat log directory path
vrchat_log_dir=~/.steam/steam/steamapps/compatdata/438100/pfx/drive_c/users/steamuser/AppData/LocalLow/VRChat/VRChat

# Set output directory based on script execution location
script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
if [[ "$script_dir" == */scripts ]]; then
    output_dir="$script_dir/../logs/vrchat-logs"
else
    output_dir="$script_dir/logs/vrchat-logs"
fi

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Copy all log files
log_files=$(find "$vrchat_log_dir" -name "output_log_*.txt")
copied_count=0

for log_file in $log_files; do
    cp "$log_file" "$output_dir/"
    file_name=$(basename "$log_file")
    echo "Copied '$file_name' to 'logs/vrchat-logs'."
    ((copied_count++))
done

if [ $copied_count -gt 0 ]; then
    echo "Copied $copied_count log file(s)."
else
    echo "No log files found."
fi
