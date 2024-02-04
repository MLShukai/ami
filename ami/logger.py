import yaml
import logging
import logging.config

def get_main_thread_logger(name: str) -> logging.Logger:
    return __get_thread_logger("main", name)

def get_trainer_thread_logger(name: str) -> logging.Logger:
    return __get_thread_logger("trainer", name)

def get_inference_thread_logger(name: str) -> logging.Logger:
    return __get_thread_logger("inference", name)

def __get_thread_logger(thread_name: str, module_name: str) -> logging.Logger:
    if module_name == "__main__":
        return logging.getLogger(thread_name)
    return logging.getLogger(f"{thread_name}.{module_name}")

# Initialization
with open("sample/logging.yaml", "r") as file:
    config = yaml.safe_load(file)

logging.config.dictConfig(config)
logging.captureWarnings(True)
