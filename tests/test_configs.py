"""ここではconfigファイル上からオブジェクトを正常にインスタンス化可能かテストします。"""
import logging
import subprocess
import sys

import cv2
import hydra
import pyfakewebcam
import pytest
import torch
from hydra.utils import instantiate
from pytest_mock import MockerFixture

from ami.hydra_instantiators import (
    instantiate_data_collectors,
    instantiate_models,
    instantiate_trainers,
)
from ami.omegaconf_resolvers import register_custom_resolvers
from tests.helpers import PROJECT_ROOT

logger = logging.getLogger(__name__)

register_custom_resolvers()

CONFIG_DIR = PROJECT_ROOT / "configs"
LAUNCH_CONFIG = "launch.yaml"
EXPERIMENT_CONFIG_DIR = CONFIG_DIR / "experiment"
EXPERIMENT_CONFIG_FILES = EXPERIMENT_CONFIG_DIR.glob("*.*")

IGNORE_EXPERIMENT_CONFIGS = {
    "unity_sioconv.yaml",
    "dreamer_unity.yaml",
    "i_jepa_with_dataset.yaml",
    "bool_mask_i_jepa_with_dataset.yaml",
    "world_models_sioconv_lerp_hidden_unity.yaml",
    "i_jepa_sioconv_dreamer_multi_step_unity.yaml",
    "dreamer_multi_step_imagination_unity.yaml",
}
EXPERIMENT_CONFIG_OVERRIDES = [
    [f"experiment={file.name.rsplit('.', 1)[0]}"]
    for file in EXPERIMENT_CONFIG_FILES
    if file.name not in IGNORE_EXPERIMENT_CONFIGS
]
HYDRA_OVERRIDES = [[]] + EXPERIMENT_CONFIG_OVERRIDES


@pytest.mark.parametrize("overrides", HYDRA_OVERRIDES)
def test_instantiate(overrides: list[str], mocker: MockerFixture, tmp_path):
    cap = create_fake_camera_if_not_exists()
    mocker.patch("pythonosc.udp_client.SimpleUDPClient")
    with hydra.initialize_config_dir(str(CONFIG_DIR)):
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

    cap.release()


def create_fake_camera_if_not_exists() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap

    if sys.platform == "linux":
        cmd = ["modprobe", "v4l2loopback", "devices=1"]
        return_code = subprocess.call(cmd)
        if return_code != 0:
            raise RuntimeError(
                "Failed to create virtual camera device by "
                + "'"
                + " ".join(cmd)
                + "'. return code "
                + str(return_code)
            )
        pyfakewebcam.FakeWebcam("/dev/video0", 1280, 720)
    else:
        raise OSError("Can not create virtual camera device on MacOS and Windows.")

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap
    else:
        raise RuntimeError("Can not open created camera device...")
