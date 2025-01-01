from enum import Enum


class ModelNames(str, Enum):
    IMAGE_ENCODER = "image_encoder"
    IMAGE_DECODER = "image_decoder"
    AUDIO_ENCODER = "audio_encoder"
    AUDIO_DECODER = "audio_decoder"
    MULTIMODAL_TEMPORAL_ENCODER = "multimodal_temporal_encoder"
    POLICY_VALUE = "policy_value"
    FORWARD_DYNAMICS = "forward_dynamics"
    POLICY = "policy"
    VALUE = "value"
    I_JEPA_PREDICTOR = "i_jepa_predictor"
    I_JEPA_CONTEXT_ENCODER = "i_jepa_context_encoder"
    I_JEPA_TARGET_ENCODER = "i_jepa_target_encoder"  # Always alias to other key: `IMAGE_ENCODDER`.
    I_JEPA_CONTEXT_VISUALIZATION_DECODER = "i_jepa_context_visualization_decoder"
    I_JEPA_TARGET_VISUALIZATION_DECODER = "i_jepa_target_visualization_decoder"
    AUDIO_JEPA_PREDICTOR = "audio_jepa_predictor"
    AUDIO_JEPA_CONTEXT_ENCODER = "audio_jepa_context_encoder"
    AUDIO_JEPA_TARGET_ENCODER = "audio_jepa_target_encoder"  # Always alias to other key: `AUDIO_ENCODDER`.
    HIFIGAN_CONTEXT_AURALIZATION_GENERATOR = "hifigan_context_auralization_generator"
    HIFIGAN_TARGET_AURALIZATION_GENERATOR = "hifigan_target_auralization_generator"
