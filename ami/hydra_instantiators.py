import copy
from typing import Any

import hydra
from omegaconf import DictConfig, ListConfig, open_dict

from .data.buffers.buffer_names import BufferNames
from .data.utils import BaseDataBuffer, DataCollectorsDict, ThreadSafeDataCollector
from .logger import get_main_thread_logger
from .models.instantiations import (
    InstantiationReturnType as ModelInstantiationReturnType,
)
from .models.model_names import ModelNames
from .models.utils import ModelWrapper, ModelWrappersDict
from .trainers.utils import BaseTrainer, TrainersList

logger = get_main_thread_logger(__name__)


def instantiate_data_collectors(data_collectors_cfg: DictConfig | None) -> DataCollectorsDict:
    d = DataCollectorsDict()
    if data_collectors_cfg is None:
        logger.info("No data collector configs are provided.")
        return d

    for name, cfg in data_collectors_cfg.items():
        logger.info(f"Instantiating <{cfg._target_}>")
        collector: BaseDataBuffer | ThreadSafeDataCollector[Any] = hydra.utils.instantiate(cfg)

        if isinstance(collector, BaseDataBuffer):
            collector = ThreadSafeDataCollector(collector)

        logger.info(f"Instatiated <{collector.__class__.__name__}[{collector._buffer.__class__.__name__}]>")

        d[BufferNames(str(name))] = collector
    return d


def instantiate_models(models_cfg: DictConfig | None) -> ModelWrappersDict:
    """Models (ModelWrappersDict)をインスタンス化する。

    `model_cfg`がNoneであれば、空のModelWrappersDictを返す。

    * Instantiation Versionについて
        `_version_` を`models_cfg`に追加することでインスタンス化方法のバージョンを切り替えることができる。
        指定がない場合はデフォルトで 1 である。

        * version 1の場合のconfig 構造
            ```yaml
            <model_name>:
              _target_: path.to.model_wrapper
              default_device: ...
              model:
                _target_: path.to.model.class
                arg1: ...
            ...
            ```

        * version 2 の場合の config 構造
            ```yaml
            _version_: 2

            _target_: ami.models.instantiations.<target>
            arg1: ...
            ```
            インスタンス化関数の記述ルールに関しては `ami/models/instantiations.py`を参照
    """
    if models_cfg is None:
        logger.info("No model configs are provided.")
        return ModelWrappersDict()

    version = 1
    if "_version_" in models_cfg:
        models_cfg = copy.copy(models_cfg)
        with open_dict(models_cfg):
            version = models_cfg.pop("_version_")

    match version:
        case 1:
            logger.info("Model Instantiation Version is 1.")
            return instantiate_models_v1(models_cfg)
        case 2:
            logger.info("Model Instantiation Version is 2.")
            return instantiate_models_v2(models_cfg)
        case _:
            raise ValueError(f"Specified instantiation version is invalid: {version!r}")


def instantiate_models_v1(models_cfg: DictConfig) -> ModelWrappersDict:
    d = ModelWrappersDict()
    aliases = []
    for name, cfg_or_alias_target in models_cfg.items():
        match cfg_or_alias_target:
            case DictConfig():
                cfg = cfg_or_alias_target
                logger.info(f"Instantiating {name!r}: <{cfg._target_}[{cfg.model._target_}]>")
                model_wrapper: ModelWrapper[Any] = hydra.utils.instantiate(cfg)
                d[ModelNames(str(name))] = model_wrapper
            case str():
                target = cfg_or_alias_target
                logger.info(f"Model name {name!r} is alias to {target!r}")
                aliases.append((name, target))
            case _:
                raise RuntimeError(f"Model config or alias must be DictConfig or str! {cfg}")
    for alias, target in aliases:
        d[ModelNames(str(alias))] = d[ModelNames(str(target))]
    return d


def instantiate_models_v2(model_cfg: DictConfig) -> ModelWrappersDict:
    d = ModelWrappersDict()
    logger.info(f"Instantiating models <{model_cfg._target_}>")
    models: ModelInstantiationReturnType = hydra.utils.instantiate(model_cfg)

    aliases = []
    for name, model_wrapper_or_alias in models.items():
        match model_wrapper_or_alias:
            case ModelWrapper():
                model_wrapper = model_wrapper_or_alias
                logger.info(
                    f"Instantiated model {name!r}: <{model_wrapper.__class__.__name__}[{model_wrapper.model.__class__.__name__}]>"
                )
                d[name] = model_wrapper
            case ModelNames():
                target = model_wrapper_or_alias
                logger.info(f"Model name {name!r} is alias to {target!r}")
                aliases.append((name, target))
            case _:
                raise ValueError(f"Invalid model wrapper or alias type!: {model_wrapper_or_alias!r}")

    for alias, target in aliases:
        d[alias] = d[target]
    return d


def instantiate_trainers(trainers_cfg: DictConfig | None) -> TrainersList:
    tl = TrainersList()

    if trainers_cfg is None:
        logger.info("No trainer configs are provided.")
        return tl

    for i, (name, cfg) in enumerate(trainers_cfg.items()):
        logger.info(f"Instatiating Trainer[{i}] {name!r}: <{cfg._target_}>")
        trainer: BaseTrainer = hydra.utils.instantiate(cfg)
        trainer.name = str(name)
        tl.append(trainer)

    return tl
