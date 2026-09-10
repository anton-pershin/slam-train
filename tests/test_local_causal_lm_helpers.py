"""Shared helpers for the slow e2e test: tiny in-process model + source JSONL."""

import json
from pathlib import Path


def build_tiny_model_dir(model_dir: Path) -> str:
    """Build a tiny Llama causal LM with a chat-template tokenizer in-process."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

    model_dir.mkdir(parents=True, exist_ok=True)

    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "hello": 3,
        "world": 4,
        "the": 5,
        "sky": 6,
        "is": 7,
        "blue": 8,
    }
    tok_raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tok_raw.pre_tokenizer = pre_tokenizers.Whitespace()
    bos, eos = tok_raw.token_to_id("<s>"), tok_raw.token_to_id("</s>")
    tok_raw.post_processor = processors.TemplateProcessing(
        single="<s>:0 $A:0 </s>:1",
        pair="$A:0 </s>:1 $B:1 </s>:1",
        special_tokens=[("<s>", bos), ("</s>", eos)],
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_raw,
        unk_token="<pad>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tok.chat_template = (
        "{% for message in messages %}"
        "<s>{{ message['role'] }}: {{ message['content'] }}"
        "{% endfor %}<s>assistant: "
    )
    tok.save_pretrained(model_dir)

    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(model_dir)
    return str(model_dir)


def make_source_jsonl(path: Path, n: int) -> str:
    """Write a MergeQuality-schema JSONL with n examples; return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
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
        lines.append(json.dumps(payload, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
