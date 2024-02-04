import logging
import logging.config

import yaml


def get_main_thread_logger(name: str) -> logging.Logger:
    return __get_thread_logger("main", name)


def get_trainer_thread_logger(name: str) -> logging.Logger:
    return __get_thread_logger("trainer", name)


def get_inference_thread_logger(name: str) -> logging.Logger:
    return __get_thread_logger("inference", name)


def __get_thread_logger(thread_name: str, module_name: str) -> logging.Logger:
    if not module_name:
        raise ValueError("Module name is empty")

    # ドット始まり・終わり・途中のドットが2つ以上はエラー
    if module_name.startswith(".") or module_name.endswith(".") or ".." in module_name:
        raise ValueError(f'Invalid module name: "{module_name}"')

# Initialization
with open("sample/logging.yaml") as file:
    config = yaml.safe_load(file)

logging.config.dictConfig(config)
logging.captureWarnings(True)
