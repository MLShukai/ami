"""Launch script file for the ami system."""
import hydra
import rootutils
from omegaconf import DictConfig

from ami.data.utils import DataCollectorsDict
from ami.interactions.interaction import Interaction
from ami.logger import get_main_thread_logger
from ami.models.utils import ModelWrappersDict
from ami.omegaconf_resolvers import register_custom_resolvers
from ami.threads import (
    InferenceThread,
    MainThread,
    TrainingThread,
    attach_shared_objects_pool_to_threads,
)
from ami.trainers.utils import TrainersList
from ami.hydra_instantiators import instantiate_data_collectors, instantiate_models, instantiate_trainers

# Add the project root path to environment vartiable `PROJECT_ROOT`
# to refer in the config file by `${oc.env:PROJECT_ROOT}`.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logger = get_main_thread_logger(__name__)

# カスタム config resolverを追加
register_custom_resolvers()


@hydra.main(config_path="../configs", config_name="launch", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info("Launch AMI.")

    logger.info(f"Instantiating Interaction <{cfg.interaction._target_}>")
    interaction: Interaction = hydra.utils.instantiate(cfg.interaction)

    logger.info(f"Instantiating DataCollectors...")
    data_collectors: DataCollectorsDict = instantiate_data_collectors(cfg.data_collectors)

    logger.info(f"Instantiating Models...")
    models: ModelWrappersDict = instantiate_models(cfg.models)
    models.send_to_default_device()

    logger.info(f"Instantiating Trainers...")
    trainers: TrainersList = instantiate_trainers(cfg.trainers)

    logger.info("Instantiating Thread Classes...")
    threads_cfg = cfg.threads
    logger.info(f"Instantiating MainThread: <{threads_cfg.main_thread._target_}>")
    main_thread: MainThread = hydra.utils.instantiate(threads_cfg.main_thread)

    logger.info(f"Instantiating InferenceThread: <{threads_cfg.inference_thread._target_}>")
    inference_thread: InferenceThread = hydra.utils.instantiate(
        threads_cfg.inference_thread, interaction=interaction, data_collectors=data_collectors
    )

    logger.info(f"Instantiating TrainingThread: <{threads_cfg.training_thread._target_}>")
    training_thread: TrainingThread = hydra.utils.instantiate(
        threads_cfg.training_thread, trainers=trainers, models=models
    )

    logger.info("Sharing objects...")
    attach_shared_objects_pool_to_threads(main_thread, inference_thread, training_thread)

    logger.info("Starting threads.")

    inference_thread.start()
    training_thread.start()

    main_thread.run()  # Shutdown 命令または Ctrl+Cが送信されるまでブロッキング

    inference_thread.join()
    training_thread.join()

    logger.info("Terminated AMI.")


if __name__ == "__main__":
    main()
