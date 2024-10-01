import pytest
from omegaconf import OmegaConf

from ami.hydra_instantiators import (
    ModelNames,
    ModelWrapper,
    instantiate_models,
    instantiate_models_v1,
    instantiate_models_v2,
)


def test_instantiate_models_v1():
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
    models = instantiate_models_v1(cfg)
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
    models = instantiate_models_v1(cfg)
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
        instantiate_models_v1(cfg)


def instantiate_func_for_v2(height, width, channels, latent_dim):
    from ami.models.vae import Conv2dEncoder

    return {
        ModelNames.I_JEPA_CONTEXT_ENCODER: ModelNames.IMAGE_ENCODER,  # Alias
        ModelNames.IMAGE_ENCODER: ModelWrapper(Conv2dEncoder(height, width, channels, latent_dim)),
    }


def test_instantiate_models_v2():
    # Test instantiate
    cfg = OmegaConf.create(
        """\
        _target_: tests.test_hydra_instantiators.instantiate_func_for_v2
        height: 84
        width: 84
        channels: 3
        latent_dim: 512
        """
    )

    models = instantiate_models_v2(cfg)
    assert ModelNames.IMAGE_ENCODER in models
    assert ModelNames.I_JEPA_CONTEXT_ENCODER in models
    assert models[ModelNames.IMAGE_ENCODER] is models[ModelNames.I_JEPA_CONTEXT_ENCODER]


def test_instantiate_models():

    # No config
    models = instantiate_models(None)
    assert len(models.items()) == 0

    # Test instantiate v1
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

    # Test instantiate v2
    cfg = OmegaConf.create(
        """
        # version 2
        _target_: tests.test_hydra_instantiators.instantiate_func_for_v2
        height: 84
        width: 84
        channels: 3
        latent_dim: 512
        """
    )

    models = instantiate_models(cfg)
    assert ModelNames.IMAGE_ENCODER in models
