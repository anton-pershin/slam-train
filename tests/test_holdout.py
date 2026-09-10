"""Tests for holdout publication (KISS spec 03, T16-T23)."""

import json

import pytest
from omegaconf import DictConfig, OmegaConf
from slam_core.collections.text_generation import MergeQuality

from slam_train.data.holdout import (
    holdout_collection_name,
    publish_collection_config,
    split_hash,
    write_holdout_jsonl,
)
from slam_train.data.sft_dataset import materialize_collection, split_indices


def _make_source_line(i: int) -> str:
    payload = {
        "ground_truth": {
            "unique_identifiers": {"name": f"Person {i}"},
            "attributes": {"a": i},
        },
        "provided_identifiers": {"name": f"Person {i}"},
        "chunks": [
            {
                "format": "json",
                "owner_id": "target",
                "content": json.dumps({"a": i, "name": f"Person {i}"}),
            }
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


@pytest.fixture
def source_jsonl(tmp_path):
    path = tmp_path / "source.jsonl"
    lines = [_make_source_line(i) for i in range(10)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, lines


@pytest.fixture
def source_collection_cfg(source_jsonl):
    path, _ = source_jsonl
    cfg = OmegaConf.create(
        {
            "_target_": "slam_core.collections.text_generation.MergeQuality",
            "name": "test__merge_quality",
            "jsonl_path": str(path),
            "user_prompt_template": "IDs: {unique_identifiers}\nChunks: {data_chunks}",
        }
    )
    return cfg


class TestWriteHoldoutJsonl:
    def test_writes_one_line_per_held_out_example(self, tmp_path, source_jsonl):
        _, lines = source_jsonl
        _, eval_idx = split_indices(len(lines), 0.8, seed=42)
        out = tmp_path / "holdout.jsonl"
        write_holdout_jsonl([lines[i] for i in eval_idx], out)
        written = out.read_text(encoding="utf-8").splitlines()
        assert len(written) == len(eval_idx)
        assert set(written) == {lines[i] for i in eval_idx}

    def test_written_jsonl_loadable_by_merge_quality(self, tmp_path, source_jsonl):
        _, lines = source_jsonl
        _, eval_idx = split_indices(len(lines), 0.8, seed=42)
        out = tmp_path / "holdout.jsonl"
        write_holdout_jsonl([lines[i] for i in eval_idx], out)

        collection = MergeQuality(
            name="holdout",
            jsonl_path=str(out),
            user_prompt_template="IDs: {unique_identifiers}\nChunks: {data_chunks}",
        )
        cases = materialize_collection(collection)
        assert len(cases) == len(eval_idx)
        # Each case parses and yields the attributes from the source line.
        for case, idx in zip(cases, sorted(eval_idx)):
            source_payload = json.loads(lines[idx])
            assert case["y_true"] == source_payload["ground_truth"]["attributes"]


class TestSplitHash:
    def test_hash_is_short_hex(self):
        assert len(split_hash(0.8, 42)) == 8
        int(split_hash(0.8, 42), 16)  # parses as hex

    def test_hash_differs_for_ratio(self):
        assert split_hash(0.8, 42) != split_hash(0.7, 42)

    def test_hash_differs_for_seed(self):
        assert split_hash(0.8, 42) != split_hash(0.8, 43)

    def test_hash_stable_for_same_params(self):
        assert split_hash(0.8, 42) == split_hash(0.8, 42)


class TestPublishCollectionConfig:
    def test_config_written_under_collection_tree(
        self, source_collection_cfg, tmp_path, source_jsonl
    ):
        _, lines = source_jsonl
        shared_root = tmp_path / "shared"
        holdout_path = tmp_path / "holdout.jsonl"
        write_holdout_jsonl(lines[:2], holdout_path)

        published = publish_collection_config(
            source_collection_cfg, holdout_path, shared_root, hash_suffix="abcd1234"
        )
        assert (
            published
            == shared_root
            / "collection"
            / "test"
            / "merge_quality_holdout_abcd1234.yaml"
        )
        assert published.exists()

    def test_config_differs_only_in_jsonl_path_and_name(
        self, source_collection_cfg, tmp_path, source_jsonl
    ):
        _, lines = source_jsonl
        shared_root = tmp_path / "shared"
        holdout_path = tmp_path / "holdout.jsonl"
        write_holdout_jsonl(lines[:2], holdout_path)

        published = publish_collection_config(
            source_collection_cfg, holdout_path, shared_root, hash_suffix="abcd1234"
        )
        generated = OmegaConf.load(published)
        assert generated._target_ == source_collection_cfg._target_
        assert (
            generated.user_prompt_template == source_collection_cfg.user_prompt_template
        )
        assert generated.jsonl_path == str(holdout_path)
        assert generated.name == "test__merge_quality_holdout_abcd1234"

    def test_idempotent_rerun_overwrites(
        self, source_collection_cfg, tmp_path, source_jsonl
    ):
        _, lines = source_jsonl
        shared_root = tmp_path / "shared"
        holdout_path = tmp_path / "holdout.jsonl"
        write_holdout_jsonl(lines[:2], holdout_path)

        first = publish_collection_config(
            source_collection_cfg, holdout_path, shared_root, hash_suffix="abcd1234"
        )
        second = publish_collection_config(
            source_collection_cfg, holdout_path, shared_root, hash_suffix="abcd1234"
        )
        assert first == second
        assert len(list((shared_root / "collection").rglob("*.yaml"))) == 1

    def test_no_jsonl_path_raises(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "_target_": "slam_core.collections.text_generation.BigBenchHard",
                "name": "bbh",
                "dataset_name": "bbh",
                "split": "train",
                "subset": "x",
                "user_prompt_template": "{original_input}",
            }
        )
        with pytest.raises(ValueError, match="jsonl_path"):
            publish_collection_config(
                cfg, tmp_path / "holdout.jsonl", tmp_path, hash_suffix="x"
            )
        # No partial artifacts.
        assert not (tmp_path / "collection").exists()


class TestHoldoutCollectionName:
    def test_name_carries_hash_suffix(self):
        assert holdout_collection_name("a__b", "deadbeef") == "a__b_holdout_deadbeef"
