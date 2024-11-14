# Run large size
python scripts/launch.py \
    experiment=bool_mask_i_jepa_with_videos \
    max_uptime=86400 \
    models=bool_mask_i_jepa_large \
    hydra.mode=MULTIRUN \
    hydra.launcher.n_jobs=1 \
    time_scale=1.4,1.4,1.4 # Run 3 times

# Run medium size
python scripts/launch.py \
    experiment=bool_mask_i_jepa_with_videos \
    max_uptime=86400 \
    models=bool_mask_i_jepa_medium \
    hydra.mode=MULTIRUN \
    hydra.launcher.n_jobs=2 \
    time_scale=0.9,0.9,1.8 # Run 3 times

# Run small size
python scripts/launch.py \
    experiment=bool_mask_i_jepa_with_videos \
    max_uptime=86400 \
    models=bool_mask_i_jepa_small \
    hydra.mode=MULTIRUN \
    hydra.launcher.n_jobs=3 \
    time_scale=0.7,0.7,0.7 # Run 3 times
