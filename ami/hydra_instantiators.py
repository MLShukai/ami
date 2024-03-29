from typing import Any

import hydra
from omegaconf import DictConfig, ListConfig

from .data.buffers.buffer_names import BufferNames
from .data.utils import BaseDataBuffer, DataCollectorsDict, ThreadSafeDataCollector
from .logger import get_main_thread_logger
from .models.model_names import ModelNames
from .models.utils import ModelWrappersDict
from .trainers.utils import TrainersList

logger = get_main_thread_logger(__name__)


def instantiate_data_collectors(data_collectors_cfg: DictConfig) -> DataCollectorsDict:
    d = DataCollectorsDict()
    for name, cfg in data_collectors_cfg.items():
        logger.info(f"Instantiating DataCollector <{cfg._target_}>")
        buffer_or_collector: BaseDataBuffer | ThreadSafeDataCollector[Any] = hydra.utils.instantiate(cfg)

        match buffer_or_collector:
            case BaseDataBuffer():
                collector = ThreadSafeDataCollector(buffer_or_collector)
            case ThreadSafeDataCollector():
                collector = buffer_or_collector
            case _:
                raise TypeError(f"Instatiated {cfg._target_} must be a DataBuffer or DataCollector!")

        d[BufferNames(str(name))] = collector
    return d
