"""LoRA SFT training on an EvalCaseCollection, via trl.SFTTrainer."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from slam_train.data.holdout import (
    publish_collection_config,
    split_hash,
    write_holdout_jsonl,
)
from slam_train.data.sft_dataset import (
    build_records,
    materialize_collection,
    split_cases,
    split_indices,
)
from slam_train.utils.common import get_config_path

CONFIG_NAME = "config_train_sft"

LOGGER = logging.getLogger(__name__)


def _read_raw_lines(jsonl_path: str) -> list[str]:
    return [
        line
        for line in Path(jsonl_path)
        .expanduser()
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def build_lora_config(lora_cfg: DictConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules),
    )


def publish_holdout(
    collection_cfg: DictConfig,
    raw_lines: list[str],
    train_ratio: float,
    seed: int,
    output_dir: Path,
    slam_shared_config: str,
) -> Path:
    """Write the holdout JSONL and publish its collection config to the registry."""
    hash_suffix = split_hash(train_ratio, seed)
    _, eval_idx = split_indices(len(raw_lines), train_ratio, seed)
    holdout_lines = [raw_lines[i] for i in eval_idx]

    holdout_jsonl = output_dir / f"holdout_{hash_suffix}.jsonl"
    write_holdout_jsonl(holdout_lines, holdout_jsonl)

    return publish_collection_config(
        source_collection_cfg=collection_cfg,
        holdout_jsonl_path=holdout_jsonl,
        shared_config_root=Path(slam_shared_config).expanduser(),
        hash_suffix=hash_suffix,
    )


def main(cfg: DictConfig) -> None:
    if "jsonl_path" not in cfg.collection:
        raise ValueError(
            "Training requires a file-backed collection with a 'jsonl_path' "
            f"field; the configured collection "
            f"{cfg.collection.get('name', '?')} has none"
        )

    output_dir = Path(cfg.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    collection = hydra.utils.instantiate(cfg.collection)
    cases = materialize_collection(collection)
    train_cases, eval_cases = split_cases(cases, cfg.split.train_ratio, cfg.split.seed)

    LOGGER.info(
        "Training on %s cases, holding out %s", len(train_cases), len(eval_cases)
    )

    if eval_cases:
        raw_lines = _read_raw_lines(cfg.collection.jsonl_path)
        if len(raw_lines) != len(cases):
            raise ValueError(
                f"Source JSONL has {len(raw_lines)} non-empty lines but the "
                f"collection yielded {len(cases)} cases"
            )
        published_config = publish_holdout(
            collection_cfg=cfg.collection,
            raw_lines=raw_lines,
            train_ratio=cfg.split.train_ratio,
            seed=cfg.split.seed,
            output_dir=output_dir,
            slam_shared_config=cfg.slam_shared_config,
        )
        LOGGER.info("Published holdout collection config to %s", published_config)

    train_records = build_records(train_cases)

    import datasets
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer_path = cfg.get("tokenizer_path") or cfg.base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model_path)

    train_dataset = datasets.Dataset.from_list(train_records)
    peft_config = build_lora_config(cfg.lora)

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        use_cpu=cfg.sft.use_cpu,
        max_length=cfg.sft.max_length,
        num_train_epochs=cfg.sft.num_train_epochs,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        learning_rate=cfg.sft.learning_rate,
        logging_steps=cfg.sft.logging_steps,
        save_strategy=cfg.sft.save_strategy,
        report_to=list(cfg.sft.report_to),
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    resolved_cfg_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(cfg, resolved_cfg_path)
    LOGGER.info("Saved adapter, tokenizer, and config to %s", output_dir)


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(main)()
