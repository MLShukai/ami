import logging


def get_main_thread_logger(name: str) -> logging.Logger:
    return __get_thread_logger("main", name)


def get_training_thread_logger(name: str) -> logging.Logger:
    return __get_thread_logger("training", name)


def get_inference_thread_logger(name: str) -> logging.Logger:
    return __get_thread_logger("inference", name)


def __get_thread_logger(thread_name: str, module_name: str) -> logging.Logger:
    if not module_name:
        raise ValueError("Module name is empty")

    # ドット始まり・終わり・途中のドットが2つ以上はエラー
    if module_name.startswith(".") or module_name.endswith(".") or ".." in module_name:
        raise ValueError(f'Invalid module name: "{module_name}"')

    return logging.getLogger(f"{thread_name}.{module_name}")
