"""Stage 2: QLoRA fine-tuning.

Loads Qwen2.5-3B-Instruct in 4-bit NF4, wraps it with LoRA adapters, and
fine-tunes on the Alpaca-style data from stage 1. Needs a CUDA GPU - run
this on Colab via notebooks/colab_runner.ipynb, not locally.

Uses plain transformers.Trainer rather than trl's SFTTrainer: we're already
building the PEFT model ourselves (below), so SFTTrainer's main value-adds
(auto-wrapping a model with LoRA, auto-tokenizing a text field) don't buy us
anything - and trl's SFTTrainer/SFTConfig API has changed shape several
times across releases, which is exactly the kind of dependency churn worth
avoiding when a stable, plain Trainer does the same job. See
PaddedCompletionCollator below for the loss-masking logic that used to come
from trl's DataCollatorForCompletionOnlyLM.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import find_subsequence, load_config, resolve_path  # noqa: E402

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU visible. QLoRA needs a GPU - run this on Colab "
            "(Runtime > Change runtime type > T4 GPU), not locally."
        )


def build_model_and_tokenizer(cfg):
    model_cfg = cfg["model"]
    compute_dtype = getattr(torch, model_cfg["bnb_4bit_compute_dtype"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg["load_in_4bit"],
        bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=model_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_id"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    return model, tokenizer


def build_peft_model(model, cfg, training_cfg):
    # Standard QLoRA prep: casts norm layers to fp32 for stability and makes
    # sure the (frozen) input embeddings require grad, which gradient
    # checkpointing needs to backprop through a 4-bit frozen base at all.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training_cfg["gradient_checkpointing"]
    )

    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def tokenize_and_mask(example, tokenizer, response_ids, max_seq_len):
    """Tokenize the pre-templated text and mask everything up to and
    including the assistant-turn opener with -100, so cross-entropy loss is
    only computed on the assistant's own tokens - not the instruction."""
    input_ids = tokenizer(
        example["text"], add_special_tokens=False, truncation=True, max_length=max_seq_len
    )["input_ids"]

    start = find_subsequence(input_ids, response_ids)
    labels = list(input_ids)
    if start is None:
        labels = [-100] * len(labels)
    else:
        response_start = start + len(response_ids)
        labels[:response_start] = [-100] * response_start

    return {"input_ids": input_ids, "labels": labels}


class PaddedCompletionCollator:
    """Pads a batch of already-masked (input_ids, labels) pairs to the
    longest sequence in the batch. Padding positions get attention_mask=0
    and label=-100, so they never contribute to loss or attention."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        max_len = max(len(ex["input_ids"]) for ex in examples)
        input_ids, attention_mask, labels = [], [], []
        for ex in examples:
            ids, labs = ex["input_ids"], ex["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(labs + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    require_cuda()
    cfg = load_config()
    training_cfg = cfg["training"]
    data_cfg = cfg["data"]

    model, tokenizer = build_model_and_tokenizer(cfg)
    model = build_peft_model(model, cfg, training_cfg)

    response_ids = tokenizer(RESPONSE_TEMPLATE, add_special_tokens=False)["input_ids"]

    train_ds = load_dataset(
        "json", data_files=str(resolve_path(data_cfg["train_path"])), split="train"
    )
    val_ds = load_dataset(
        "json", data_files=str(resolve_path(data_cfg["val_path"])), split="train"
    )

    map_fn = lambda ex: tokenize_and_mask(  # noqa: E731
        ex, tokenizer, response_ids, data_cfg["max_seq_len"]
    )
    train_ds = train_ds.map(map_fn, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(map_fn, remove_columns=val_ds.column_names)

    # Periodic training-time eval only needs to be big enough to show the
    # loss trend, not to be a precise quality estimate - that's what
    # evaluate.py's full-val-set perplexity check is for. Subsetting here
    # keeps eval_steps pauses short instead of scanning all of val_ds every
    # time.
    eval_subset_size = min(training_cfg.get("eval_subset_size", len(val_ds)), len(val_ds))
    train_time_eval_ds = val_ds.select(range(eval_subset_size))

    collator = PaddedCompletionCollator(pad_token_id=tokenizer.pad_token_id)

    output_dir = resolve_path(training_cfg["output_dir"])
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_ratio=training_cfg["warmup_ratio"],
        weight_decay=training_cfg["weight_decay"],
        logging_steps=training_cfg["logging_steps"],
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg["save_total_limit"],
        eval_strategy="steps",
        eval_steps=training_cfg["eval_steps"],
        fp16=training_cfg["fp16"],
        optim=training_cfg["optim"],
        gradient_checkpointing=training_cfg["gradient_checkpointing"],
        seed=training_cfg["seed"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=train_time_eval_ds,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=_find_latest_checkpoint(output_dir))

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    log_path = output_dir / "train_log_history.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"Saved adapter + tokenizer to {output_dir}")
    print(f"Saved training log history to {log_path}")


def _find_latest_checkpoint(output_dir: Path):
    """Resume automatically if a Colab disconnect left checkpoints behind."""
    if not output_dir.exists():
        return None
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
    )
    return str(checkpoints[-1]) if checkpoints else None


if __name__ == "__main__":
    main()
