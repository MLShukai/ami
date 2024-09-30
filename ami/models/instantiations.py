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
from typing import Any, TypeAlias

import torch

from .model_names import ModelNames
from .model_wrapper import ModelWrapper

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
