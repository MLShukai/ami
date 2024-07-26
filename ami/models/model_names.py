from enum import Enum


class ModelNames(str, Enum):
    IMAGE_ENCODER = "image_encoder"
    IMAGE_DECODER = "image_decoder"
    POLICY_VALUE = "policy_value"
    FORWARD_DYNAMICS = "forward_dynamics"
    POLICY = "policy"
    VALUE = "value"
