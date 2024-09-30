gammas=(0.5 0.7 0.8 0.9)

# Simple World
for g in "${gammas[@]}"
do
    python scripts/launch.py experiment=i_jepa_sioconv_ppo_multi_step \
        max_uptime=18000 \
        interaction/environment=unity \
        interaction.environment.file_path=/workspace/unity_executables/SimpleWorld/SimpleWorld.x86_64 \
        models=i_jepa_sioconv_resnetpolicy_small \
        hydra.mode=MULTIRUN \
        hydra.launcher.n_jobs=1 \
        data_collectors.ppo_trajectory.gamma=$g \
        time_scale=4,4,4 # Run 3 times
done

# Noisy World
for g in "${gammas[@]}"
do
    python scripts/launch.py experiment=i_jepa_sioconv_ppo_multi_step \
        max_uptime=18000 \
        interaction/environment=unity \
        interaction.environment.file_path=/workspace/unity_executables/MeshiTeroNoisyWorld/MeshiTeroNoisyWorld.x86_64 \
        models=i_jepa_sioconv_resnetpolicy_small \
        hydra.mode=MULTIRUN \
        hydra.launcher.n_jobs=1 \
        data_collectors.ppo_trajectory.gamma=$g \
        time_scale=4,4,4 # Run 3 times
done
