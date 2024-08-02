#!/bin/bash

# Set default values
TOTAL_RUNS=10
PARALLEL_RUNS=$(( $(nproc) / 4 ))
COMMAND=""

# Help function
show_help() {
    echo "Usage: $0 [-n total_runs] [-p parallel_runs] [-h] command"
    echo "  -n: Total number of runs (default: 10)"
    echo "  -p: Number of parallel runs (default: 1/4 of CPU cores)"
    echo "  -h: Display this help message"
    exit 1
}

# Process command line options
while getopts "n:p:h" opt; do
    case $opt in
        n) TOTAL_RUNS=$OPTARG ;;
        p) PARALLEL_RUNS=$OPTARG ;;
        h) show_help ;;
        \?) echo "Invalid option: -$OPTARG" >&2; show_help ;;
    esac
done

# Remove option arguments and treat the rest as the command
shift $((OPTIND-1))
COMMAND="$@"

# Error if no command is specified
if [ -z "$COMMAND" ]; then
    echo "Error: No command specified." >&2
    show_help
fi

# Execution function
run_command() {
    $COMMAND
}

# Parallel execution
export -f run_command
seq $TOTAL_RUNS | xargs -P $PARALLEL_RUNS -I {} bash -c 'run_command'

echo "All executions completed."
