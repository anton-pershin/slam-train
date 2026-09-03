import os
from pathlib import Path


def get_project_path() -> Path:
    return Path(__file__).parent.parent.parent


def get_config_path() -> Path:
    return get_project_path() / "config"


def set_cuda_visible_devices(devices: list[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))

