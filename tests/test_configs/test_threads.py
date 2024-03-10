import hydra
import pytest

from tests.helpers import PROJECT_ROOT

INTERACTION_CONFIG_DIR = PROJECT_ROOT / "configs" / "threads"


@pytest.mark.parametrize("config_name", ["default"])
def test_instantiate(config_name):
    with hydra.initialize_config_dir(str(INTERACTION_CONFIG_DIR)):

        hydra.utils.instantiate(hydra.compose(config_name))
