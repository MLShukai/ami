"""Launch script file for the ami system."""
import hydra
import rootutils
from omegaconf import DictConfig

from ami.logger import get_main_thread_logger

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logger = get_main_thread_logger(__name__)


@hydra.main(config_path="../configs", config_name="launch", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info("Launch AMI.")


if __name__ == "__main__":
    main()
