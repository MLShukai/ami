#!/bin/bash

# Available task patterns
declare -a tasks=("whitenoize" "blackout" "flickerimage")

# Display available tasks
echo "Available tasks:"
for i in "${!tasks[@]}"; do
    echo "  $i: ${tasks[$i]}"
done

# Get task selection from user
while true; do
    read -p "Select task number (0-$((${#tasks[@]}-1))): " task_num

    # Validate input is a number
    if ! [[ "$task_num" =~ ^[0-9]+$ ]]; then
        echo "Please enter a valid number"
        continue
    fi

    # Validate input is in range
    if [ "$task_num" -lt 0 ] || [ "$task_num" -ge "${#tasks[@]}" ]; then
        echo "Please enter a number between 0 and $((${#tasks[@]}-1))"
        continue
    fi

    break
done

# Set task name from selection
task_name="${tasks[$task_num]}"
echo "Selected task: $task_name"

# Define parameter arrays
gamma_values=(0.9 0.97 0.991)
entropy_coef_values=(0.1 0.04 0.016)

# Outer loop for multiple runs
for run in {1..3}; do
    echo "Starting run $run of 3"

    # Initialize grid pattern index
    i=0

    # Nested loops for parameter combinations
    for gamma in "${gamma_values[@]}"; do
        for entropy_coef in "${entropy_coef_values[@]}"; do
            echo "Running with gamma=$gamma, entropy_coef=$entropy_coef (pattern $i)"

            python scripts/launch.py \
                experiment=temporal/${task_name}_no_action \
                models.multimodal_temporal_encoder.inference_forward.path=ami.models.temporal_encoder.inference_forward_with_layernorm \
                data_collectors.ppo_trajectory.gamma=$gamma \
                trainers.ppo.entropy_coef=$entropy_coef \
                time_scale=3.2 \
                task_tag=entcoef_gamma_search.${i}

            # Increment the pattern index
            i=$((i + 1))
        done
    done
done
