"""Stage 3: evaluation - base model vs fine-tuned model.

Two complementary checks, because neither one alone tells the full story:

1. Masked perplexity on held-out val data - a single number, cheap to
   compute, good for tracking whether fine-tuning helped at all. But a model
   can have great perplexity while still giving unhelpful or off-topic
   answers (perplexity only measures how "expected" the reference tokens
   were, not whether a *different* good answer would have scored well too).
2. Qualitative side-by-side generations on a fixed prompt set - slower to
   judge, but it's the only way to actually see behavioral differences
   (formatting, tone, following instructions) that perplexity can miss.

Needs a CUDA GPU - run this on Colab.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    build_prompt_text,
    compute_masked_perplexity,
    load_config,
    resolve_path,
)

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_base(cfg, device):
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
        model_cfg["base_model_id"], quantization_config=bnb_config, device_map=device
    )
    return model, tokenizer


def load_finetuned(base_model, adapter_dir):
    return PeftModel.from_pretrained(base_model, adapter_dir)


@torch.no_grad()
def generate(model, tokenizer, prompt_text, device, max_new_tokens=256):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    device = "cuda" if torch.cuda.is_available() else None
    if device is None:
        raise RuntimeError("No CUDA GPU visible - run this on Colab.")

    cfg = load_config()
    eval_cfg = cfg["eval"]
    adapter_dir = str(resolve_path(cfg["training"]["output_dir"]))

    val_ds = load_dataset(
        "json", data_files=str(resolve_path(cfg["data"]["val_path"])), split="train"
    )
    ppl_examples = list(val_ds.select(range(min(eval_cfg["n_perplexity_examples"], len(val_ds)))))

    print("Loading base model (4-bit) ...")
    base_model, tokenizer = load_base(cfg, device)

    print("Scoring base-model perplexity on held-out set ...")
    base_ppl = compute_masked_perplexity(base_model, tokenizer, ppl_examples, device=device)
    print(f"Base perplexity (response tokens only): {base_ppl:.3f}")

    print("Generating base-model qualitative outputs ...")
    base_outputs = []
    for prompt in eval_cfg["qualitative_prompts"]:
        prompt_text = build_prompt_text(tokenizer, {"instruction": prompt, "input": ""})
        base_outputs.append(generate(base_model, tokenizer, prompt_text, device))

    print(f"Attaching fine-tuned adapter from {adapter_dir} ...")
    ft_model = load_finetuned(base_model, adapter_dir)

    print("Scoring fine-tuned-model perplexity on held-out set ...")
    ft_ppl = compute_masked_perplexity(ft_model, tokenizer, ppl_examples, device=device)
    print(f"Fine-tuned perplexity (response tokens only): {ft_ppl:.3f}")

    print("Generating fine-tuned-model qualitative outputs ...")
    ft_outputs = []
    for prompt in eval_cfg["qualitative_prompts"]:
        prompt_text = build_prompt_text(tokenizer, {"instruction": prompt, "input": ""})
        ft_outputs.append(generate(ft_model, tokenizer, prompt_text, device))

    report = {
        "perplexity": {"base": base_ppl, "fine_tuned": ft_ppl},
        "qualitative": [
            {"prompt": p, "base": b, "fine_tuned": f}
            for p, b, f in zip(eval_cfg["qualitative_prompts"], base_outputs, ft_outputs)
        ],
    }

    report_path = resolve_path("outputs/benchmarks/eval_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved eval report to {report_path}")
    print("\n=== Perplexity summary ===")
    print(f"Base:       {base_ppl:.3f}")
    print(f"Fine-tuned: {ft_ppl:.3f}  ({'better' if ft_ppl < base_ppl else 'worse'} than base)")
    print("\n=== Sample qualitative comparison ===")
    for item in report["qualitative"][:2]:
        print(f"\nPrompt: {item['prompt']}")
        print(f"  Base:       {item['base'][:200]}")
        print(f"  Fine-tuned: {item['fine_tuned'][:200]}")


if __name__ == "__main__":
    main()
