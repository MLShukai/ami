"""ここではconfigファイル上からオブジェクトを正常にインスタンス化可能かテストします。"""
import hydra
import pytest
from hydra.utils import instantiate

from tests.helpers import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "configs"
LAUNCH_CONFIG = "launch.yaml"
EXPERIMENT_CONFIG_DIR = CONFIG_DIR / "experiment"
EXPERIMENT_CONFIG_FILES = EXPERIMENT_CONFIG_DIR.glob("*.*")

EXPERIMENT_CONFIG_OVERRIDES = [[f"experiment={file.name.rsplit('.', 1)[0]}"] for file in EXPERIMENT_CONFIG_FILES]
HYDRA_OVERRIDES = [[]] + EXPERIMENT_CONFIG_OVERRIDES


@pytest.mark.parametrize("overrides", HYDRA_OVERRIDES)
def test_instantiate(overrides: list[str]):
    with hydra.initialize_config_dir(str(CONFIG_DIR)):
        cfg = hydra.compose(LAUNCH_CONFIG, overrides=overrides)

        interaction = instantiate(cfg.interaction)
        data_collectors = instantiate(cfg.data_collectors)
        models = instantiate(cfg.models)
        trainers = instantiate(cfg.trainers)

        threads = cfg.threads
        instantiate(threads.main_thread)
        instantiate(threads.inference_thread, interaction=interaction, data_collectors=data_collectors)
        instantiate(threads.training_thread, models=models, trainers=trainers)
