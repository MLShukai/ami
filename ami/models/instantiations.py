"""Modelをインスタンス化する関数を記述するファイルです。

いくつかのルールがあります。
* インスタンス化対象のモデルは 関数のローカルスコープに記述する
* 関数名とconfig名は一致させる
* 関数の戻り値の型は常に `InstantiationReturnType`であること。

辞書のValueに`ModelNames`を指定した場合、それがAliasキーとして機能します。

---
This file describes functions for instantiating Models.

There are several rules:
* The model to be instantiated should be described in the local scope of the function
* The function name and config name should match
* The return type of the function should always be `InstantiationReturnType`

If you specify `ModelNames` as the value of a dictionary, it will function as an Alias key.
"""
import copy
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
import torch.nn as nn

from .model_names import ModelNames
from .model_wrapper import ModelWrapper
from .utils import size_2d, size_2d_to_int_tuple

InstantiationReturnType: TypeAlias = dict[ModelNames, ModelWrapper[Any] | ModelNames]

# For default value.
CPU_DEVICE = torch.device("cpu")


def image_vae(
    device: torch.device = CPU_DEVICE,
    height: int = 84,
    width: int = 84,
    channels: int = 3,
    latent_dim: int = 512,
) -> InstantiationReturnType:
    from .vae import Conv2dDecoder, Conv2dEncoder, encoder_infer

    return {
        ModelNames.IMAGE_ENCODER: ModelWrapper(
            model=Conv2dEncoder(
                height,
                width,
                channels,
                latent_dim,
            ),
            default_device=device,
            has_inference=True,
            inference_forward=encoder_infer,
        ),
        ModelNames.IMAGE_DECODER: ModelWrapper(
            model=Conv2dDecoder(
                height,
                width,
                channels,
                latent_dim,
            ),
            default_device=device,
            has_inference=False,
        ),
    }


def i_jepa_mean_patch_sioconv_resnetpolicy(
    device: torch.device = CPU_DEVICE,
    img_size: size_2d = (144, 144),  # (height, width)
    in_channels: int = 3,
    patch_size: size_2d = (12, 12),  # (height, width)
    action_choices_per_category: list[int] | None = None,
    i_jepa_encoder: dict[str, Any] = {},
    i_jepa_predictor: dict[str, Any] = {},
    sioconv: dict[str, Any] = {},
    policy_value: dict[str, Any] = {},
) -> InstantiationReturnType:

    # Defining nested dict config.
    @dataclass
    class IJEPAEncoderConfig:
        embed_dim: int = 216
        out_dim: int = 216
        depth: int = 6
        num_heads: int = 3
        mlp_ratio: float = 4.0

    @dataclass
    class IJEPAPredictorConfig:
        hidden_dim: int = 108
        depth: int = 6
        num_heads: int = 2

    @dataclass
    class SioConvConfig:
        action_embedding_dim: int = 8
        depth: int = 6
        dim: int = 1024
        num_head: int = 4
        chunk_size: int = 512
        dropout: float = 0.1

    @dataclass
    class PolicyValueConfig:
        dim: int = 512
        dim_hidden: int = 512
        depth: int = 4

    i_jepa_encoder_cfg = IJEPAEncoderConfig(**i_jepa_encoder)
    i_jepa_predictor_cfg = IJEPAPredictorConfig(**i_jepa_predictor)
    sioconv_cfg = SioConvConfig(**sioconv)
    policy_value_cfg = PolicyValueConfig(**policy_value)

    # import dependencies
    from ami.interactions.environments.actuators.vrchat_osc_discrete_actuator import (
        ACTION_CHOICES_PER_CATEGORY,
    )

    from .bool_mask_i_jepa import (
        BoolMaskIJEPAEncoder,
        BoolTargetIJEPAPredictor,
        encoder_infer_mean_patch,
    )
    from .components.discrete_policy_head import DiscretePolicyHead
    from .components.fully_connected_fixed_std_normal import (
        FullyConnectedFixedStdNormal,
    )
    from .components.fully_connected_value_head import FullyConnectedValueHead
    from .components.multi_embeddings import MultiEmbeddings
    from .components.resnet import ResNetFF
    from .components.sioconv import SioConv
    from .forward_dynamics import ForwardDynamcisWithActionReward
    from .policy_value_common_net import (
        ConcatFlattenedObservationAndLerpedHidden,
        LerpStackedHidden,
        PolicyValueCommonNet,
    )

    img_size = size_2d_to_int_tuple(img_size)
    patch_size = size_2d_to_int_tuple(patch_size)

    n_patch_vertical, r = divmod(img_size[0], patch_size[0])
    assert r == 0
    n_patch_horizontal, r = divmod(img_size[1], patch_size[1])
    assert r == 0
    n_patches = (n_patch_vertical, n_patch_horizontal)

    if action_choices_per_category is None:
        action_choices_per_category = ACTION_CHOICES_PER_CATEGORY

    i_jepa_encoder_model = BoolMaskIJEPAEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=i_jepa_encoder_cfg.embed_dim,
        out_dim=i_jepa_encoder_cfg.out_dim,
        depth=i_jepa_encoder_cfg.depth,
        num_heads=i_jepa_encoder_cfg.num_heads,
        mlp_ratio=i_jepa_encoder_cfg.mlp_ratio,
    )

    return {
        ModelNames.IMAGE_ENCODER: ModelNames.I_JEPA_TARGET_ENCODER,  # Alias for agent.
        ModelNames.I_JEPA_TARGET_ENCODER: ModelWrapper(
            default_device=device,
            has_inference=True,
            inference_forward=encoder_infer_mean_patch,
            model=i_jepa_encoder_model,
        ),
        ModelNames.I_JEPA_CONTEXT_ENCODER: ModelWrapper(
            default_device=device,
            has_inference=False,
            model=copy.deepcopy(i_jepa_encoder_model),
        ),
        ModelNames.I_JEPA_PREDICTOR: ModelWrapper(
            default_device=device,
            has_inference=False,
            model=BoolTargetIJEPAPredictor(
                n_patches=n_patches,
                context_encoder_out_dim=i_jepa_encoder_cfg.out_dim,
                hidden_dim=i_jepa_predictor_cfg.hidden_dim,
                depth=i_jepa_predictor_cfg.depth,
                num_heads=i_jepa_predictor_cfg.num_heads,
            ),
        ),
        ModelNames.FORWARD_DYNAMICS: ModelWrapper(
            default_device=device,
            has_inference=True,
            model=ForwardDynamcisWithActionReward(
                observation_flatten=nn.Identity(),
                action_flatten=MultiEmbeddings(
                    choices_per_category=action_choices_per_category,
                    embedding_dim=sioconv_cfg.action_embedding_dim,
                    do_flatten=True,
                ),
                obs_action_projection=nn.Linear(
                    in_features=len(action_choices_per_category) * sioconv_cfg.action_embedding_dim
                    + i_jepa_encoder_cfg.out_dim,
                    out_features=sioconv_cfg.dim,
                ),
                core_model=SioConv(
                    depth=sioconv_cfg.depth,
                    dim=sioconv_cfg.dim,
                    num_head=sioconv_cfg.num_head,
                    dim_ff_hidden=sioconv_cfg.dim * 2,
                    chunk_size=sioconv_cfg.chunk_size,
                    dropout=sioconv_cfg.dropout,
                ),
                obs_hat_dist_head=FullyConnectedFixedStdNormal(
                    dim_in=sioconv_cfg.dim, dim_out=i_jepa_encoder_cfg.out_dim, normal_cls="Deterministic"
                ),
                action_hat_dist_head=DiscretePolicyHead(
                    dim_in=sioconv_cfg.dim,
                    action_choices_per_category=action_choices_per_category,
                ),
                reward_hat_dist_head=FullyConnectedFixedStdNormal(
                    dim_in=sioconv_cfg.dim, dim_out=1, squeeze_feature_dim=True
                ),
            ),
        ),
        ModelNames.POLICY_VALUE: ModelWrapper(
            default_device=device,
            has_inference=True,
            model=PolicyValueCommonNet(
                observation_projection=nn.Identity(),
                forward_dynamics_hidden_projection=LerpStackedHidden(
                    dim=sioconv_cfg.dim,
                    depth=sioconv_cfg.depth,
                    num_head=sioconv_cfg.num_head,
                ),
                observation_hidden_projection=ConcatFlattenedObservationAndLerpedHidden(
                    dim_obs=i_jepa_encoder_cfg.out_dim,
                    dim_hidden=sioconv_cfg.dim,
                    dim_out=policy_value_cfg.dim,
                ),
                core_model=ResNetFF(
                    dim=policy_value_cfg.dim,
                    dim_hidden=policy_value_cfg.dim_hidden,
                    depth=policy_value_cfg.depth,
                ),
                policy_head=DiscretePolicyHead(
                    dim_in=policy_value_cfg.dim,
                    action_choices_per_category=action_choices_per_category,
                ),
                value_head=FullyConnectedValueHead(
                    dim_in=policy_value_cfg.dim,
                    squeeze_value_dim=True,
                ),
            ),
        ),
    }
