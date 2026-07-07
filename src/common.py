"""Shared helpers used across every stage: config loading, the Alpaca-record
-> chat-message formatting used by data prep/eval/benchmarking alike, and
the masked-perplexity metric shared by evaluate.py and benchmark.py (kept in
one place so all three stay in sync)."""

import math
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "configs/qlora_3b.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    """Config paths are written relative to the repo root regardless of cwd."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def alpaca_to_messages(example: dict, include_output: bool = True) -> list[dict]:
    """Turn one yahma/alpaca-cleaned record into chat-template messages.

    Alpaca records have `instruction`, an optional `input`, and `output`.
    When `input` is present it's appended after the instruction rather than
    treated as a separate turn - that's how the original Alpaca prompt
    format works, and it's what the base model's own instruction-tuning
    would have seen for this dataset family.
    """
    instruction = example["instruction"].strip()
    extra_input = (example.get("input") or "").strip()
    user_content = f"{instruction}\n\n{extra_input}" if extra_input else instruction

    messages = [{"role": "user", "content": user_content}]
    if include_output and "output" in example:
        messages.append({"role": "assistant", "content": example["output"].strip()})
    return messages


def build_prompt_text(tokenizer, example: dict) -> str:
    """Chat-templated prompt for generation - instruction/input only, with
    the trailing assistant-turn opener the model should continue from."""
    messages = alpaca_to_messages(example, include_output=False)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def find_subsequence(haystack: list, needle: list):
    """First index where `needle` occurs in `haystack`, or None. Used to
    locate the assistant-turn boundary in a tokenized chat-template string
    so training (train.py) and evaluation (compute_masked_perplexity below)
    mask the prompt out of the loss identically."""
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def compute_masked_perplexity(
    model,
    tokenizer,
    examples: list,
    device: str = "cuda",
    response_template: str = "<|im_start|>assistant\n",
) -> float:
    """Perplexity computed only over assistant-response tokens, mirroring the
    loss masking used in training (see train.py's PaddedCompletionCollator).
    Scoring the whole templated prompt+response would conflate "predicts the
    user's own instruction back" with "writes a good answer" - two very
    different things - so only the response tokens count.

    Aggregated as exp(total_nll / total_tokens) across the whole example set,
    NOT a mean of per-example perplexities: examples have different response
    lengths, and averaging perplexity values directly over-weights short
    responses relative to their actual token contribution.
    """
    response_ids = tokenizer(response_template, add_special_tokens=False)["input_ids"]
    total_nll = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for ex in examples:
            text = tokenizer.apply_chat_template(
                alpaca_to_messages(ex, include_output=True), tokenize=False
            )
            input_ids = tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)

            start = find_subsequence(input_ids[0].tolist(), response_ids)
            if start is None:
                continue
            response_start = start + len(response_ids)

            labels = input_ids.clone()
            labels[:, :response_start] = -100
            n_response_tokens = int((labels != -100).sum().item())
            if n_response_tokens == 0:
                continue

            outputs = model(input_ids=input_ids, labels=labels)
            total_nll += outputs.loss.item() * n_response_tokens
            total_tokens += n_response_tokens

    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)
