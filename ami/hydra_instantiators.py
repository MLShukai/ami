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


def instantiate_data_collectors(data_collectors_cfg: DictConfig) -> DataCollectorsDict:
    d = DataCollectorsDict()
    for name, cfg in data_collectors_cfg.items():
        logger.info(f"Instantiating <{cfg._target_}>")
        collector: BaseDataBuffer | ThreadSafeDataCollector[Any] = hydra.utils.instantiate(cfg)

        if isinstance(collector, BaseDataBuffer):
            collector = ThreadSafeDataCollector(collector)

        logger.info(f"Instatiated <{collector.__class__.__name__}[{collector._buffer.__class__.__name__}]>")

        d[BufferNames(str(name))] = collector
    return d


def instantiate_models(models_cfg: DictConfig) -> ModelWrappersDict:
    d = ModelWrappersDict()
    for name, cfg in models_cfg.items():
        logger.info(f"Instantiating <{cfg._target_}[{cfg.model._target_}]>")
        model_wrapper: ModelWrapper[Any] = hydra.utils.instantiate(cfg)

        d[ModelNames(str(name))] = model_wrapper
    return d


def instantiate_trainers(trainers_cfg: DictConfig) -> TrainersList:
    tl = TrainersList()
    for i, (name, cfg) in enumerate(trainers_cfg.items()):
        logger.info(f"Instatiating Trainer[{i}] {name!r}: <{cfg._target_}>")
        trainer: BaseTrainer = hydra.utils.instantiate(cfg)
        tl.append(trainer)

    return tl
