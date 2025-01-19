"""This test checks whether object can be successfully instantiated from config
file."""
import logging
import shutil
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
import pytest
from hydra.utils import instantiate
from pytest_mock import MockerFixture

from ami.hydra_instantiators import (
    instantiate_data_collectors,
    instantiate_models,
    instantiate_trainers,
)
from ami.omegaconf_resolvers import register_custom_resolvers
from tests.helpers import PROJECT_ROOT

register_custom_resolvers()

CONFIG_DIR = PROJECT_ROOT / "configs"
LAUNCH_CONFIG = "launch.yaml"
EXPERIMENT_CONFIG_DIR = CONFIG_DIR / "experiment"
EXPERIMENT_CONFIG_FILES = EXPERIMENT_CONFIG_DIR.glob("**/*.*")

IGNORE_EXPERIMENT_CONFIGS = {
    "unity_sioconv.yaml",
    "dreamer_unity.yaml",
    "i_jepa_with_dataset.yaml",
    "bool_mask_i_jepa_with_dataset.yaml",
    "world_models_sioconv_lerp_hidden_unity.yaml",
    "i_jepa_sioconv_dreamer_multi_step_unity.yaml",
    "dreamer_multi_step_imagination_unity.yaml",
}

DATA_DIR = PROJECT_ROOT / "data"
if not (DATA_DIR / "random_observation_action_log").exists():
    IGNORE_EXPERIMENT_CONFIGS.add("bool_mask_i_jepa.yaml")
    IGNORE_EXPERIMENT_CONFIGS.add("bool_mask_i_jepa_with_videos.yaml")
    IGNORE_EXPERIMENT_CONFIGS.add("learn_only_sioconv.yaml")
    IGNORE_EXPERIMENT_CONFIGS.add("learn_i_jepa_sioconv.yaml")
    IGNORE_EXPERIMENT_CONFIGS.add("i_jepa_mean_patch_with_videos.yaml")

if not (DATA_DIR / "2024-09-14_09-42-23,678417.ckpt").exists():
    IGNORE_EXPERIMENT_CONFIGS.add("learn_only_sioconv.yaml")

if not (DATA_DIR / "japan_street_images"):
    IGNORE_EXPERIMENT_CONFIGS.add("i_jepa_sioconv_ppo_fundamental_flickerimage.yaml")
    IGNORE_EXPERIMENT_CONFIGS.add("multimodal_temporal/flickerimage_no_action.yaml")

try:
    import soundcard
except Exception:
    IGNORE_EXPERIMENT_CONFIGS.add("random_observation_action_log.yaml")
    IGNORE_EXPERIMENT_CONFIGS.add("multimodal/vrchat.yaml")


def _get_experiment_name(path: Path) -> str:
    name = str(path)[len(str(EXPERIMENT_CONFIG_DIR)) + 1 :]
    return name.rsplit(".", 1)[0]


EXPERIMENT_CONFIG_OVERRIDES = [
    [f"experiment={_get_experiment_name(file)}"]
    for file in EXPERIMENT_CONFIG_FILES
    if file.name not in IGNORE_EXPERIMENT_CONFIGS
]
HYDRA_OVERRIDES = [[]] + EXPERIMENT_CONFIG_OVERRIDES

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("overrides", HYDRA_OVERRIDES)
def test_instantiate(overrides: list[str], mocker: MockerFixture, tmp_path, caplog):
    conditional_video_capture_mock(mocker)
    mocker.patch("pythonosc.udp_client.SimpleUDPClient")
    with hydra.initialize_config_dir(str(CONFIG_DIR)):
        logger.info(overrides)
        cfg = hydra.compose(LAUNCH_CONFIG, overrides=overrides + ["devices=cpu"], return_hydra_config=True)
        cfg.paths.output_dir = tmp_path

        interaction = instantiate(cfg.interaction)
        data_collectors = instantiate_data_collectors(cfg.data_collectors)
        models = instantiate_models(cfg.models)
        trainers = instantiate_trainers(cfg.trainers)
        checkpoint_scheduler = instantiate(cfg.checkpointing)

        threads = cfg.threads
        instantiate(threads.main_thread, checkpoint_scheduler=checkpoint_scheduler)
        instantiate(threads.inference_thread, interaction=interaction, data_collectors=data_collectors)
        instantiate(threads.training_thread, models=models, trainers=trainers)


def conditional_video_capture_mock(mocker: MockerFixture):
    original_video_capture = cv2.VideoCapture  # avoid mock reference.

    def mock_video_capture(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], str):
            # If the first argument is a string (likely a file path), use the original VideoCapture
            return original_video_capture(*args, **kwargs)
        else:
            # For other cases (e.g., camera index), return a mock object
            mock = mocker.Mock(spec=original_video_capture)
            mock.isOpened.return_value = True
            mock.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock.get.return_value = 1.0
            return mock

    return mocker.patch("cv2.VideoCapture", side_effect=mock_video_capture)
