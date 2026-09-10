"""Config composition tests (KISS spec 03, T31-T32)."""

from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra")

from hydra import compose, initialize  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

PROJECT_PATH = Path(__file__).parent.parent
SLAM_CORE_PATH = PROJECT_PATH.parent / "slam-core"
SLAM_EVAL_PATH = PROJECT_PATH.parent / "slam-eval"


@pytest.fixture
def shared_config_env(monkeypatch):
    monkeypatch.setenv("SLAM_SHARED_CONFIG", str(SLAM_CORE_PATH / "config"))


class TestConfigTrainSft:
    def test_composes_with_shared_collection_via_searchpath(self, shared_config_env):
        with initialize(
            version_base="1.3",
            config_path="../config",
            job_name="test_train_sft",
        ):
            cfg = compose(config_name="config_train_sft")

        assert (
            cfg.collection._target_
            == "slam_core.collections.text_generation.MergeQuality"
        )
        assert cfg.collection.name == "merge_quality__merge_quality_easy"
        assert cfg.split.train_ratio == 0.8
        assert cfg.split.seed == 42
        assert cfg.slam_shared_config == str(SLAM_CORE_PATH / "config")

    def test_searchpath_block_structurally_identical_to_slam_eval(self):
        # Compare the raw YAML text: OmegaConf resolves interpolations on access.
        train_text = (PROJECT_PATH / "config" / "config_train_sft.yaml").read_text()
        eval_text = (SLAM_EVAL_PATH / "config" / "config_main.yaml").read_text()

        def searchpath_block(text: str) -> list[str]:
            lines = text.splitlines()
            start = next(i for i, l in enumerate(lines) if l.strip() == "searchpath:")
            entries = []
            for line in lines[start + 1 :]:
                if not line.strip().startswith("- "):
                    break
                entries.append(line.strip())
            return entries

        assert searchpath_block(train_text) == searchpath_block(eval_text)
