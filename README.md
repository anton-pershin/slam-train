# slam-train

SFT and training library

## Getting started

1. Create a virtual environment, e.g.
```bash
conda create -n myenv python=3.12
conda activate myenv
```
2. Install necessary packages
```bash
pip install -r requirements.txt
# For local development against a local slam-core checkout:
# pip install -e ../slam-core
```
3. Set up `/config/user_settings/user_settings.yaml`
4. Run the training script and override the corresponding config file in `/config/config_train_sft.yaml`
```bash
python slam_train/scripts/train_sft.py
```

⚠️  DO NOT commit your `user_settings.yaml`

## Scripts

### `train_sft.py`

Runs LoRA supervised fine-tuning via `trl.SFTTrainer` on an eval case
collection resolved from the shared slam config tree
(`slam-core/config`, via `hydra.searchpath` and `SLAM_SHARED_CONFIG`).

The collection is materialized in full, shuffled with a configured seed, and
split into disjoint train/eval subsets by `split.train_ratio`. Training records
are built as `trl` prompt-completion pairs: the prompt is the collection's
`user_prompt` (plus a system message when the collection emits one), the
completion is the ground truth — `json.dumps(y_true, ensure_ascii=False,
indent=2)` for non-string targets (e.g. `MergeQuality` dicts), which matches
what `MergeQualityScorer.safe_parse_prediction` accepts at eval time.

#### Configuration

1. In `user_settings.yaml`, set up your paths:
   ```yaml
   model_root: /home/tony/models
   dataset_root: /home/tony/datasets
   project_path: /path/to/slam-train
   ```
   `output_dir` defaults to the hydra run directory.

2. In `config_train_sft.yaml`, modify:
   - `collection` — which shared collection to train on (must be file-backed,
     i.e. have a `jsonl_path`)
   - `base_model_path` — the base causal LM to fine-tune
   - `split.train_ratio`, `split.seed` — the train/eval split
   - `lora.*` — LoRA hyperparameters (`r`, `lora_alpha`, `lora_dropout`,
     `target_modules`; architecture-dependent)
   - `sft.*` — `SFTTrainer` arguments (epochs, batch size, learning rate,
     `max_length`)

#### Output

Writes to `output_dir`:
- the trained LoRA adapter (standard `peft` layout, loadable via
  `PeftModel.from_pretrained`) and the tokenizer
- `resolved_config.yaml` — the fully resolved training configuration
- `holdout_<hash>.jsonl` — the held-out eval examples in the source dataset
  schema, where `<hash>` is derived from `train_ratio` and `seed`
- a generated collection config for the holdout, published to the shared
  config tree under `collection/`, named `<source_name>_holdout_<hash>`

#### Follow-up evaluation

Evaluate the trained adapter with `slam-eval` against the generated holdout
collection (no code changes needed in `slam-eval`):
```bash
python slam_eval/scripts/main.py \
  collection=merge_quality__merge_quality_easy_holdout_<hash> \
  model=local_hf_causal_lm \
  model.adapter_path=/path/to/output_dir
```
`model.adapter_path: null` gives the adapter-free baseline on the same holdout.

## Tests

Fast unit tests (no `torch`):
```bash
pytest --ignore=tests/e2e
```
The slow end-to-end smoke test (tiny in-process model, CPU):
```bash
pytest tests/e2e -m slow
```