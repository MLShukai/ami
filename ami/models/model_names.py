from enum import Enum


class ModelNames(str, Enum):
    IMAGE_ENCODER = "image_encoder"
    IMAGE_DECODER = "image_decoder"
    POLICY_VALUE = "policy_value"
    FORWARD_DYNAMICS = "forward_dynamics"
    POLICY = "policy"
    VALUE = "value"
    I_JEPA_PREDICTOR = "i_jepa_predictor"
    I_JEPA_CONTEXT_ENCODER = "i_jepa_context_encoder"
    I_JEPA_TARGET_ENCODER = "i_jepa_target_encoder"  # Always alias to other key: `IMAGE_ENCODDER`.
