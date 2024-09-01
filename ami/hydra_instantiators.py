from typing import Any

import hydra
from omegaconf import DictConfig, ListConfig

from .data.buffers.buffer_names import BufferNames
from .data.utils import BaseDataBuffer, DataCollectorsDict, ThreadSafeDataCollector
from .logger import get_main_thread_logger
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
    d = ModelWrappersDict()
    if models_cfg is None:
        logger.info("No model configs are provided.")
        return d

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


def instantiate_trainers(trainers_cfg: DictConfig | None) -> TrainersList:
    tl = TrainersList()

    if trainers_cfg is None:
        logger.info("No trainer configs are provided.")
        return tl

    for i, (name, cfg) in enumerate(trainers_cfg.items()):
        logger.info(f"Instatiating Trainer[{i}] {name!r}: <{cfg._target_}>")
        trainer: BaseTrainer = hydra.utils.instantiate(cfg)
        trainer.name = name
        tl.append(trainer)

    return tl
