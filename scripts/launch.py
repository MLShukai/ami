"""Launch script file for the ami system."""
import os

import hydra
import rootutils
import torch
from omegaconf import DictConfig, open_dict

from ami.checkpointing.checkpoint_schedulers import BaseCheckpointScheduler
from ami.checkpointing.checkpointing import Checkpointing
from ami.data.utils import DataCollectorsDict
from ami.hydra_instantiators import (
    instantiate_data_collectors,
    instantiate_models,
    instantiate_trainers,
)
from ami.interactions.interaction import Interaction
from ami.logger import display_nested_config, get_main_thread_logger
from ami.models.utils import ModelWrappersDict, create_model_parameter_count_dict
from ami.omegaconf_resolvers import register_custom_resolvers
from ami.tensorboard_loggers import TensorBoardLogger
from ami.threads import (
    InferenceThread,
    MainThread,
    TrainingThread,
    attach_shared_objects_pool_to_threads,
)
from ami.trainers.utils import TrainersList

# Add the project root path to environment vartiable `PROJECT_ROOT`
# to refer in the config file by `${oc.env:PROJECT_ROOT}`.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logger = get_main_thread_logger(__name__)

# カスタム config resolverを追加
register_custom_resolvers()


@hydra.main(config_path="../configs", config_name="launch", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info("Launch AMI.")
    if precision := cfg.get("torch_float32_matmul_precision"):
        torch.set_float32_matmul_precision(precision)

    logger.info(f"Instantiating Interaction <{cfg.interaction._target_}>")
    interaction: Interaction = hydra.utils.instantiate(cfg.interaction)

    logger.info("Instantiating DataCollectors...")
    data_collectors: DataCollectorsDict = instantiate_data_collectors(cfg.data_collectors)

    logger.info("Instantiating Models...")
    models: ModelWrappersDict = instantiate_models(cfg.models)
    models.send_to_default_device()
    param_count = create_model_parameter_count_dict(models)

    logger.info("Instantiating Trainers...")
    trainers: TrainersList = instantiate_trainers(cfg.trainers)

    logger.info("Instantiating Checkpointing...")
    checkpoint_scheduler: BaseCheckpointScheduler = hydra.utils.instantiate(cfg.checkpointing)
    checkpointing = checkpoint_scheduler.checkpointing

    logger.info("Instantiating Thread Classes...")
    threads_cfg = cfg.threads
    logger.info(f"Instantiating MainThread: <{threads_cfg.main_thread._target_}>")
    main_thread: MainThread = hydra.utils.instantiate(
        threads_cfg.main_thread, checkpoint_scheduler=checkpoint_scheduler
    )

    logger.info(f"Instantiating InferenceThread: <{threads_cfg.inference_thread._target_}>")
    inference_thread: InferenceThread = hydra.utils.instantiate(
        threads_cfg.inference_thread, interaction=interaction, data_collectors=data_collectors
    )

    logger.info(f"Instantiating TrainingThread: <{threads_cfg.training_thread._target_}>")
    training_thread: TrainingThread = hydra.utils.instantiate(
        threads_cfg.training_thread, trainers=trainers, models=models
    )

    # Logging to tensorboard.
    tensorboard_logger = TensorBoardLogger(
        os.path.join(
            cfg.paths.tensorboard_dir,
            "__main__",
        )
    )
    hparams_dict = {
        "interaction": cfg.interaction,
        "data": cfg.data_collectors,
        "models": cfg.models,
        "trainers": cfg.trainers,
        "param_count": param_count,
    }
    tensorboard_logger.log_hyperparameters(hparams_dict)

    with open_dict(cfg) as c:
        c["param_count"] = param_count

    display_cfg = display_nested_config(cfg)
    logger.info(f"Displaying configs..\n{display_cfg}")
    with open(os.path.join(cfg.paths.output_dir, "launch-configuration.yaml"), "w") as f:
        f.write(display_cfg)

    logger.info("Sharing objects...")
    attach_shared_objects_pool_to_threads(main_thread, inference_thread, training_thread)

    checkpointing.add_threads(main_thread, inference_thread, training_thread)

    if (ckpt_path := cfg.get("saved_checkpoint_path", None)) is not None:
        logger.info(f"Loading the checkpoint from '{ckpt_path}'")
        checkpointing.load_checkpoint(ckpt_path)

    logger.info("Starting threads.")

    inference_thread.start()
    training_thread.start()

    main_thread.run()  # Shutdown 命令または Ctrl+Cが送信されるまでブロッキング

    inference_thread.join()
    training_thread.join()

    ckpt_path = checkpointing.save_checkpoint()

    logger.info(f"Saved the final checkpoint to '{ckpt_path}'")

    logger.info("Terminated AMI.")


if __name__ == "__main__":
    main()
