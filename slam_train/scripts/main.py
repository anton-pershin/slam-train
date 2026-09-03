from omegaconf import DictConfig
import hydra

from slam_train.utils.common import get_config_path

CONFIG_NAME = "config_main"


def main(cfg: DictConfig) -> None:
    ...


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(main)()
