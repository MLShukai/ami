from __future__ import annotations

from functools import partial
from typing import Any, TypeAlias

import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from torch.utils.data import DataLoader

from ami.models.utils import size_2d
from ami.tensorboard_loggers import StepIntervalLogger
from ami.trainers.bool_mask_i_jepa_trainer import BoolMaskIJEPATrainer
from ami.trainers.components.bool_i_jepa_mask_collator import (
    BoolIJEPAMultiBlockMaskCollator,
)

RawCfg: TypeAlias = dict[str, Any]


class TrainerCfg(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    device: str | torch.device
    max_epochs: PositiveInt = 3
    minimum_dataset_size: PositiveInt | None = None
    minimum_new_data_count: PositiveInt = 128
    target_encoder_update_moving_average: float = Field(ge=0.0, le=1.0, default=0.996)


class DataLoaderCfg(BaseModel):
    batch_size: PositiveInt = 8
    shuffle: bool = True


class CollateFnCfg(BaseModel):
    input_size: size_2d
    patch_size: size_2d
    mask_scale: tuple[float, float] = (0.025, 0.125)
    aspect_ratio: tuple[float, float] = (0.75, 1.5)
    n_masks: PositiveInt = 4
    min_keep: PositiveInt = 10


class OptimizerCfg(BaseModel):
    cls: type[torch.optim.Optimizer] = torch.optim.AdamW
    hparams: dict[str, Any] = Field(default_factory=lambda: {"lr": 1e-4, "weight_decay": 0.04})


class LoggerCfg(BaseModel):
    log_dir: str
    log_every_n_steps: PositiveInt = 1


def instantiate(
    trainer: RawCfg,
    dataloader: RawCfg,
    collate_fn: RawCfg,
    optimizer: RawCfg,
    logger: RawCfg,
) -> BoolMaskIJEPATrainer:
    trainer_cfg = TrainerCfg(**trainer)
    dl_cfg = DataLoaderCfg(**dataloader)
    cf_cfg = CollateFnCfg(**collate_fn)
    optim_cfg = OptimizerCfg(**optimizer)
    log_cfg = LoggerCfg(**logger)
    return BoolMaskIJEPATrainer(
        # DataLoader
        partial_dataloader=partial(
            DataLoader,
            batch_size=dl_cfg.batch_size,
            shuffle=dl_cfg.shuffle,
            drop_last=True,
            collate_fn=BoolIJEPAMultiBlockMaskCollator(
                cf_cfg.input_size,
                patch_size=cf_cfg.patch_size,
                mask_scale=cf_cfg.mask_scale,
                n_masks=cf_cfg.n_masks,
                aspect_ratio=cf_cfg.aspect_ratio,
                min_keep=cf_cfg.min_keep,
            ),
        ),
        # Optimizer
        partial_optimizer=partial(optim_cfg.cls, **optim_cfg.hparams),
        # Logger
        logger=StepIntervalLogger(
            log_cfg.log_dir,
            log_every_n_steps=log_cfg.log_every_n_steps,
        ),
        # Trainer
        device=torch.device(trainer_cfg.device),
        target_encoder_update_moving_average=trainer_cfg.target_encoder_update_moving_average,
        max_epochs=trainer_cfg.max_epochs,
        minimum_dataset_size=trainer_cfg.minimum_dataset_size
        if trainer_cfg.minimum_dataset_size is not None
        else dl_cfg.batch_size,
        minimum_new_data_count=trainer_cfg.minimum_new_data_count,
    )
