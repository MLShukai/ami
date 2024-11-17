from enum import Enum


class BufferNames(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"
    PPO_TRAJECTORY = "ppo_trajectory"
    FORWARD_DYNAMICS_TRAJECTORY = "forward_dynamics_trajectory"
    DREAMING_INITIAL_STATES = "dreaming_initial_states"
