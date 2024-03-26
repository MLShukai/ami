from ..step_data import DataKeys
from .causal_data_buffer import CausalDataBuffer


class ForwardDynamicsTrajectoryBuffer(CausalDataBuffer):
    def __init__(self, max_len: int):
        super().__init__(
            max_len,
            key_list=[
                DataKeys.EMBED_OBSERVATION,
                DataKeys.NEXT_EMBED_OBSERVATION,
                DataKeys.ACTION,
                DataKeys.HIDDEN,
            ],
        )
