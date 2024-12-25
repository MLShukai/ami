#!/bin/bash

# Define parameter arrays
gamma_values=(0.9 0.924 0.948 0.973 0.999)
entropy_coef_values=(0.1 0.056 0.032 0.018 0.01)

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
                experiment=i_jepa_sioconv_ppo_fundamental_blackout \
                data_collectors.ppo_trajectory.gamma=$gamma \
                trainers.ppo.entropy_coef=$entropy_coef \
                time_scale=5.0 \
                task_tag=detailed_entcoef_gamma_${i}

            # Increment the pattern index
            i=$((i + 1))
        done
    done
done
