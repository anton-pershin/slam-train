"""Publication of the training-time eval holdout to the shared config registry."""

from __future__ import annotations

import hashlib
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def split_hash(train_ratio: float, seed: int) -> str:
    """Short digest over the canonical string of all split parameters.

    Both the ratio and the seed participate: the same ratio with a different
    seed yields an entirely different holdout, so the two must never collide
    on one name.
    """
    canonical = f"ratio={train_ratio}|seed={seed}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def write_holdout_jsonl(raw_lines: list[str], jsonl_path: Path) -> None:
    """Write held-out raw JSONL lines verbatim, in the source dataset schema."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for line in raw_lines:
            handle.write(line.rstrip("\n") + "\n")


def holdout_collection_name(source_name: str, hash_suffix: str) -> str:
    """Collection name for a holdout: source name plus the split-identifying hash."""
    return f"{source_name}_holdout_{hash_suffix}"


def publish_collection_config(
    source_collection_cfg: DictConfig,
    holdout_jsonl_path: Path,
    shared_config_root: Path,
    hash_suffix: str,
) -> Path:
    """Publish a generated collection config for the holdout to the shared tree.

    Copies the source collection config verbatim, overriding only ``jsonl_path``
    and ``name``. Raises if the source config has no ``jsonl_path`` (the
    mechanism supports file-backed collections only).
    """
    if "jsonl_path" not in source_collection_cfg:
        raise ValueError(
            "Holdout publication requires a file-backed collection with a "
            f"'jsonl_path' field; collection config "
            f"{source_collection_cfg.get('name', '?')} has none"
        )

    generated = OmegaConf.merge(source_collection_cfg, {})
    generated.jsonl_path = str(holdout_jsonl_path)
    generated.name = holdout_collection_name(source_collection_cfg.name, hash_suffix)

    # A collection name encodes its config location: '/' in the config tree
    # maps to '__' in the name (e.g. merge_quality/merge_quality_easy.yaml ->
    # merge_quality__merge_quality_easy).
    config_rel_path = generated.name.replace("__", "/") + ".yaml"
    config_path = Path(shared_config_root) / "collection" / config_rel_path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(generated, config_path)

    return config_path
