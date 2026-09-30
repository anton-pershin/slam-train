"""End-to-end smoke test for train_sft (KISS spec 03, T34).

Slow: imports torch, runs SFTTrainer for 1 epoch on a tiny in-process model
on CPU. Asserts wiring (artifacts written, loadable), not learning.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("hydra")
trl = pytest.importorskip("trl")
peft = pytest.importorskip("peft")
pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")

from hydra import compose, initialize  # noqa: E402

from slam_train.scripts.train_sft import main  # noqa: E402
from tests.test_local_causal_lm_helpers import (  # noqa: E402
    build_tiny_model_dir,
    make_source_jsonl,
)

PROJECT_PATH = Path(__file__).parent.parent.parent


@pytest.mark.slow
class TestTrainSftEndToEnd:
    def test_full_run_writes_artifacts(self, tmp_path, monkeypatch):
        # 1. Tiny in-process base model.
        model_dir = build_tiny_model_dir(tmp_path / "base_model")

        # 2. ~10-example MergeQuality-style source JSONL.
        source_jsonl = make_source_jsonl(tmp_path / "source.jsonl", n=10)

        # 3. Shared config tree: hosts the generated holdout collection config
        #    and a source collection config the test can select via `collection=`.
        shared_root = tmp_path / "shared_config"
        test_collection_dir = shared_root / "collection" / "e2e_test"
        test_collection_dir.mkdir(parents=True)
        (test_collection_dir / "source.yaml").write_text(
            "_target_: slam_core.collections.text_generation.MergeQuality\n"
            "name: e2e_test__source\n"
            f"jsonl_path: {source_jsonl}\n"
            "user_prompt_template: |\n"
            "  IDs: {unique_identifiers}\n"
            "  Chunks: {data_chunks}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("SLAM_SHARED_CONFIG_PATH", str(shared_root))

        # 4. Compose the config, overriding everything to point at tmp_path.
        output_dir = tmp_path / "output"
        with initialize(
            version_base="1.3",
            config_path="../../config",
            job_name="test_train_sft",
        ):
            overrides = [
                f"base_model_path={model_dir}",
                f"output_dir={output_dir}",
                "collection=e2e_test/source",
                f"slam_shared_config={shared_root}",
                "sft.num_train_epochs=1",
                "sft.max_length=128",
                "sft.logging_steps=1",
                "sft.report_to=[]",
                "split.train_ratio=0.8",
                "split.seed=42",
                "lora.target_modules=[q_proj,v_proj]",
                "lora.r=2",
                "lora.lora_alpha=4",
                "hydra.run.dir=" + str(tmp_path / "hydra_run"),
            ]
            cfg = compose(config_name="config_train_sft", overrides=overrides)

        monkeypatch.chdir(tmp_path)
        main(cfg)

        # 5. Artifacts: adapter loadable by peft, tokenizer, resolved config,
        #    holdout JSONL, generated collection config.
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter_model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(str(model_dir)),
            str(output_dir),
        )
        assert adapter_model is not None
        assert (output_dir / "adapter_model.safetensors").exists()
        AutoTokenizer.from_pretrained(str(output_dir))
        assert (output_dir / "resolved_config.yaml").exists()

        # 5b. Generated collection config published to the shared registry.
        generated_configs = [
            p
            for p in (shared_root / "collection").rglob("*.yaml")
            if p.name.startswith("source_holdout_")
        ]
        assert len(generated_configs) == 1
        # 6. Holdout JSONL in the source schema: 20% of 10 = 2 lines.
        holdout_files = list(output_dir.glob("holdout_*.jsonl"))
        assert len(holdout_files) == 1
        holdout_lines = holdout_files[0].read_text(encoding="utf-8").splitlines()
        assert len(holdout_lines) == 2

        for line in holdout_lines:
            json.loads(line)  # source-schema: each line is a valid payload

        # 7. Re-run into the same output dir overwrites idempotently.
        with initialize(
            version_base="1.3",
            config_path="../../config",
            job_name="test_train_sft",
        ):
            cfg2 = compose(config_name="config_train_sft", overrides=overrides)
        main(cfg2)
        assert (
            len(
                [
                    p
                    for p in (shared_root / "collection").rglob("*.yaml")
                    if p.name.startswith("source_holdout_")
                ]
            )
            == 1
        )
        assert len(list(output_dir.glob("holdout_*.jsonl"))) == 1
