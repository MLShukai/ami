import pytest
from omegaconf import OmegaConf

from ami.hydra_instantiators import instantiate_models
from ami.models.model_names import ModelNames


def test_instantiate_models():
    # Test instantiate
    cfg = OmegaConf.create(
        """\
        image_decoder:
          _target_: ami.models.model_wrapper.ModelWrapper
          model:
            _target_: ami.models.vae.Conv2dDecoder
            height: 84
            width: 84
            channels: 3
            latent_dim: 512
        """
    )
    models = instantiate_models(cfg)
    assert ModelNames.IMAGE_DECODER in models

    # Test alias
    cfg = OmegaConf.create(
        """\
        image_decoder:
          _target_: ami.models.model_wrapper.ModelWrapper
          model:
            _target_: ami.models.vae.Conv2dDecoder
            height: 84
            width: 84
            channels: 3
            latent_dim: 512
        image_encoder: image_decoder
        """
    )
    models = instantiate_models(cfg)
    assert ModelNames.IMAGE_DECODER in models
    assert ModelNames.IMAGE_ENCODER in models
    assert models[ModelNames.IMAGE_ENCODER] is models[ModelNames.IMAGE_DECODER]

    # Test invalid config format
    cfg = OmegaConf.create(
        """\
        image_decoder:
            _target_: ami.models.model_wrapper.ModelWrapper
            model:
              _target_: ami.models.vae.Conv2dDecoder
              height: 84
              width: 84
              channels: 3
              latent_dim: 512
        image_encoder: 0 # <- not str value is allowed!
        """
    )
    with pytest.raises(RuntimeError):
        instantiate_models(cfg)
