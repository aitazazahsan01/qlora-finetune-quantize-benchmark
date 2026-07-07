"""Stage 1: dataset prep.

Loads yahma/alpaca-cleaned (a deduped, error-corrected version of the
original 52k Alpaca set), applies an extra dedup/quality pass, formats every
record through Qwen2.5's chat template, drops anything that would blow past
max_seq_len (dropped rather than truncated - truncating an answer mid-sentence
would teach the model that trailing off is normal), and writes train/val
jsonl splits.

CPU-only - no GPU needed. Run from the repo root: `python src/data.py`
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import alpaca_to_messages, load_config, resolve_path  # noqa: E402

from datasets import load_dataset
from transformers import AutoTokenizer


def dedupe_and_clean(dataset):
    seen = set()
    keep_indices = []
    for i, ex in enumerate(dataset):
        output = (ex.get("output") or "").strip()
        if not output:
            continue
        key = (ex["instruction"].strip(), (ex.get("input") or "").strip(), output)
        if key in seen:
            continue
        seen.add(key)
        keep_indices.append(i)
    return dataset.select(keep_indices)


def write_jsonl(path: Path, examples: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    cfg = load_config()
    data_cfg = cfg["data"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_id"])

    print(f"Loading {data_cfg['raw_dataset_id']} ...")
    ds = load_dataset(data_cfg["raw_dataset_id"], split="train")
    print(f"Raw examples: {len(ds)}")

    ds = dedupe_and_clean(ds)
    print(f"After dedup + empty-output filter: {len(ds)}")

    ds = ds.shuffle(seed=data_cfg["seed"])

    n_needed = data_cfg["n_train"] + data_cfg["n_val"]
    max_seq_len = data_cfg["max_seq_len"]

    formatted = []
    dropped_too_long = 0
    for ex in ds:
        messages = alpaca_to_messages(ex, include_output=True)
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tokens > max_seq_len:
            dropped_too_long += 1
            continue
        formatted.append(
            {
                "text": text,
                "instruction": ex["instruction"],
                "input": ex.get("input", ""),
                "output": ex["output"],
            }
        )
        if len(formatted) >= n_needed:
            break

    print(f"Dropped for exceeding max_seq_len={max_seq_len}: {dropped_too_long}")
    print(f"Formatted examples kept: {len(formatted)}")

    n_train, n_val = data_cfg["n_train"], data_cfg["n_val"]
    if len(formatted) < n_needed:
        print(
            f"WARNING: only found {len(formatted)} usable examples "
            f"(wanted {n_needed}). Shrinking val split proportionally."
        )
        n_val = max(1, len(formatted) // 20)
        n_train = len(formatted) - n_val

    train_examples = formatted[:n_train]
    val_examples = formatted[n_train : n_train + n_val]

    train_path = resolve_path(data_cfg["train_path"])
    val_path = resolve_path(data_cfg["val_path"])
    write_jsonl(train_path, train_examples)
    write_jsonl(val_path, val_examples)

    print(f"Wrote {len(train_examples)} train examples -> {train_path}")
    print(f"Wrote {len(val_examples)} val examples -> {val_path}")

    print("\nSample formatted example:")
    print("-" * 60)
    print(train_examples[0]["text"])
    print("-" * 60)


if __name__ == "__main__":
    main()
