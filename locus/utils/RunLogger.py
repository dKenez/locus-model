import logging
from pathlib import Path
from typing import Iterable, Union

logging.basicConfig(level=logging.INFO, handlers=[])


def RunLogger(log_paths: Iterable[Union[Path, str]]):
    runLogger = logging.getLogger("runLogger")

    for handler in runLogger.handlers:
        runLogger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for log_path in log_paths:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        runLogger.addHandler(file_handler)

    return runLogger
