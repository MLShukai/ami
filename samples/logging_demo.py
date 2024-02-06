import logging
import logging.config

import yaml

from ami.logger import (
    get_inference_thread_logger,
    get_main_thread_logger,
    get_training_thread_logger,
)

with open("samples/logging.yaml") as file:
    config = yaml.safe_load(file)

logging.config.dictConfig(config)


main_logger = get_main_thread_logger("very_very_long_module_name")
training_logger = get_training_thread_logger("very_very_long_module_name")
inference_logger = get_inference_thread_logger("very_very_long_module_name")

main_logger.debug("This is a debug message")
training_logger.info("This is an info message")
inference_logger.warning("This is a warning message")

try:
    get_main_thread_logger("")
except ValueError as e:
    print(e)

try:
    get_main_thread_logger(".demo")
except ValueError as e:
    print(e)

try:
    get_main_thread_logger("demo.")
except ValueError as e:
    print(e)

try:
    get_main_thread_logger("try..demo")
except ValueError as e:
    print(e)
