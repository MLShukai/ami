from enum import StrEnum


class BufferNames(StrEnum):
    IMAGE = "image"
    PPO_TRAJECTORY = "ppo_trajectory"
    FORWARD_DYNAMICS_TRAJECTORY = "forward_dynamics_trajectory"
