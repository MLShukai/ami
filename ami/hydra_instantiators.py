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
    """Instantiate Models (ModelWrappersDict).

    If `model_cfg` is None, return an empty ModelWrappersDict.

    About Instantiation Version:
        The instantiation version is switched by finding `_target_` in the root of `model_cfg`.
        If `_target_` does not exist in the root, the version is estimated to be 1; otherwise, it is 2.
        Please write the version info in your new config comment.

    Config structure for version 1:
        ```yaml
        <model_name>:
            _target_: path.to.model_wrapper
            default_device: ...
            model:
                _target_: path.to.model.class
                arg1: ...
            ...
        ```

    Config structure for version 2:
        ```yaml
        # version: 2

        _target_: path.to.instantiation_func
        arg1: ...
        ```
        For rules on describing the instantiation function, refer to `ami/models/instantiations.py`
    """
    if models_cfg is None:
        logger.info("No model configs are provided.")
        return ModelWrappersDict()

    if "_target_" not in models_cfg:
        logger.info("Model Instantiation Version is 1.")
        return instantiate_models_v1(models_cfg)
    else:
        logger.info("Model Instantiation Version is 2.")
        return instantiate_models_v2(models_cfg)


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
                    f"Instantiated model '{name}': <{model_wrapper.__class__.__name__}[{model_wrapper.model.__class__.__name__}]>"
                )
                d[name] = model_wrapper
            case ModelNames():
                target = model_wrapper_or_alias
                logger.info(f"Model name '{name}' is alias to '{target}'")
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
        if cfg is None:
            logger.info(f"Trainer[{i}] {name!r} is null. passing...")
            continue
        logger.info(f"Instatiating Trainer[{i}] {name!r}: <{cfg._target_}>")
        trainer: BaseTrainer = hydra.utils.instantiate(cfg)
        trainer.name = str(name)
        tl.append(trainer)

    return tl
