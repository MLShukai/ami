from enum import StrEnum


class ModelNames(StrEnum):
    IMAGE_ENCODER = "image_encoder"
    IMAGE_DECODER = "image_decoder"
    POLICY_VALUE = "policy_value"
    FORWARD_DYNAMICS = "forward_dynamics"
