"""Stage 5: benchmark speed vs quality across every quantization variant.

Variants compared:
  - fp16       : merged model, no quantization (the quality ceiling)
  - bnb_8bit   : bitsandbytes load-time 8-bit
  - bnb_4bit   : bitsandbytes load-time 4-bit (NF4, same scheme as training)
  - gguf_Q8_0, gguf_Q5_K_M, gguf_Q4_K_M : static pre-quantized llama.cpp files

Two different perplexity numbers show up in the report and they are NOT
directly comparable:
  - HF variants use the same masked-response perplexity from evaluate.py
    (scored only over assistant tokens, via transformers).
  - GGUF variants use llama.cpp's own `llama-perplexity` tool, which scores
    the *whole* text sequence (no prompt masking) - it's the standard tool
    for the job but measures a related, not identical, quantity. Use it to
    compare *across GGUF levels* (Q8_0 vs Q5_K_M vs Q4_K_M), not to compare
    a GGUF number against an HF number.
The qualitative generations are the one axis that's genuinely apples-to-
apples across every variant - read those side by side too, not just the
numbers.

Needs a CUDA GPU and the GGUF files from quantize.py - run this on Colab
after both train.py and quantize.py have completed.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    alpaca_to_messages,
    build_prompt_text,
    compute_masked_perplexity,
    load_config,
    resolve_path,
)

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

GEN_MAX_NEW_TOKENS = 128


# --------------------------------------------------------------------------
# HF variants: fp16 / bnb-8bit / bnb-4bit
# --------------------------------------------------------------------------


def benchmark_hf_variant(name, model, tokenizer, device, ppl_examples, prompts):
    torch.cuda.reset_peak_memory_stats(device)

    ppl = compute_masked_perplexity(model, tokenizer, ppl_examples, device=device)

    total_new_tokens = 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    for p in prompts:
        prompt_text = build_prompt_text(tokenizer, {"instruction": p, "input": ""})
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        total_new_tokens += out.shape[1] - inputs["input_ids"].shape[1]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "name": name,
        "perplexity": ppl,
        "tokens_per_sec": total_new_tokens / elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }


def run_hf_variants(cfg, merged_dir, ppl_examples, prompts, device, report, save_report):
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    variants = [
        ("fp16", None),
        ("bnb_8bit", BitsAndBytesConfig(load_in_8bit=True)),
        (
            "bnb_4bit",
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
                bnb_4bit_use_double_quant=model_cfg["bnb_4bit_use_double_quant"],
                bnb_4bit_compute_dtype=getattr(torch, model_cfg["bnb_4bit_compute_dtype"]),
            ),
        ),
    ]

    done = {r["name"] for r in report["hf_variants"]}
    for name, bnb_config in variants:
        if name in done:
            print(f"\n--- HF variant {name} already in report, skipping ---")
            continue
        print(f"\n--- Benchmarking HF variant: {name} ---")
        kwargs = {"device_map": device}
        if bnb_config is not None:
            kwargs["quantization_config"] = bnb_config
        else:
            kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(str(merged_dir), **kwargs)

        result = benchmark_hf_variant(name, model, tokenizer, device, ppl_examples, prompts)
        report["hf_variants"].append(result)
        save_report()
        print(result)

        del model
        torch.cuda.empty_cache()

    return tokenizer


# --------------------------------------------------------------------------
# GGUF variants, via llama.cpp CLI tools
# --------------------------------------------------------------------------


def prepare_gguf_perplexity_corpus(ppl_examples, tokenizer):
    corpus_path = resolve_path("outputs/benchmarks/val_corpus.txt")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for ex in ppl_examples:
            text = tokenizer.apply_chat_template(
                alpaca_to_messages(ex, include_output=True), tokenize=False
            )
            f.write(text + "\n\n")
    return corpus_path


def parse_llama_bench_tps(stdout_text):
    """llama-bench's --output json prints a JSON array of result rows; pull
    avg_ts (tokens/sec) from the text-generation row. Field names have
    shifted across llama.cpp versions - if this returns None, check the raw
    output saved alongside it in the report and adjust the key(s) here."""
    try:
        rows = json.loads(stdout_text)
        for row in rows:
            if row.get("n_gen", 0) > 0:
                return row.get("avg_ts")
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return None


def parse_llama_perplexity(stdout_text):
    """llama-perplexity prints a line like 'Final estimate: PPL = 7.12 +/- 0.05'."""
    match = re.search(r"PPL\s*=\s*([\d.]+)", stdout_text)
    return float(match.group(1)) if match else None


def run_gguf_variant(level, gguf_path, llama_bin_dir, corpus_path, sample_prompt):
    llama_cli = llama_bin_dir / "llama-cli"
    llama_perplexity = llama_bin_dir / "llama-perplexity"
    llama_bench = llama_bin_dir / "llama-bench"

    bench_proc = subprocess.run(
        [str(llama_bench), "-m", str(gguf_path), "-p", "0", "-n", "128", "-o", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    tok_per_sec = parse_llama_bench_tps(bench_proc.stdout)

    ppl_proc = subprocess.run(
        [str(llama_perplexity), "-m", str(gguf_path), "-f", str(corpus_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    ppl = parse_llama_perplexity(ppl_proc.stdout)

    gen_proc = subprocess.run(
        [str(llama_cli), "-m", str(gguf_path), "-p", sample_prompt, "-n", "128", "--no-display-prompt"],
        capture_output=True,
        text=True,
        check=True,
    )

    return {
        "name": f"gguf_{level}",
        "perplexity_gguf_unmasked": ppl,
        "tokens_per_sec": tok_per_sec,
        "file_size_gb": gguf_path.stat().st_size / 1e9,
        "sample_generation": gen_proc.stdout.strip()[:500],
        "raw_bench_stdout": bench_proc.stdout[:2000],
    }


# --------------------------------------------------------------------------


def print_summary_table(hf_results, gguf_results):
    print("\n=== Summary: speed vs quality ===")
    header = f"{'variant':<14}{'tokens/sec':>12}{'size/VRAM GB':>14}{'quality (PPL)':>16}"
    print(header)
    print("-" * len(header))
    for r in hf_results:
        print(f"{r['name']:<14}{r['tokens_per_sec']:>12.2f}{r['peak_vram_gb']:>14.2f}{r['perplexity']:>16.3f}")
    for r in gguf_results:
        tps = r["tokens_per_sec"]
        ppl = r["perplexity_gguf_unmasked"]
        print(
            f"{r['name']:<14}"
            f"{(f'{tps:.2f}' if tps is not None else 'n/a'):>12}"
            f"{r['file_size_gb']:>14.2f}"
            f"{(f'{ppl:.3f}' if ppl is not None else 'n/a'):>16}"
        )


def main():
    device = "cuda" if torch.cuda.is_available() else None
    if device is None:
        raise RuntimeError("No CUDA GPU visible - run this on Colab.")

    cfg = load_config()
    quant_cfg = cfg["quantization"]
    eval_cfg = cfg["eval"]

    merged_dir = resolve_path(quant_cfg["merged_dir"])
    gguf_dir = resolve_path(quant_cfg["gguf_dir"])
    llama_bin_dir = resolve_path(quant_cfg["llama_cpp_dir"]) / "build" / "bin"

    # Checkpoint to disk after every variant instead of only at the very end,
    # so a Colab disconnect mid-run loses at most one in-flight variant
    # instead of the whole report; reruns skip whatever's already in here.
    report_path = resolve_path("outputs/benchmarks/benchmark_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = {"hf_variants": [], "gguf_variants": []}

    def save_report():
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    val_ds = load_dataset(
        "json", data_files=str(resolve_path(cfg["data"]["val_path"])), split="train"
    )
    ppl_examples = list(val_ds.select(range(min(eval_cfg["n_perplexity_examples"], len(val_ds)))))
    prompts = eval_cfg["qualitative_prompts"]

    print("=== HF variants (fp16 / bnb-8bit / bnb-4bit) ===")
    tokenizer = run_hf_variants(cfg, merged_dir, ppl_examples, prompts, device, report, save_report)

    print("\n=== GGUF variants ===")
    corpus_path = prepare_gguf_perplexity_corpus(ppl_examples, tokenizer)
    done_gguf = {r["name"] for r in report["gguf_variants"]}
    for level in quant_cfg["gguf_levels"]:
        name = f"gguf_{level}"
        if name in done_gguf:
            print(f"{name} already in report, skipping.")
            continue
        gguf_path = gguf_dir / f"model-{level}.gguf"
        if not gguf_path.exists():
            print(f"Skipping {level}: {gguf_path} not found (did quantize.py run?)")
            continue
        print(f"\n--- Benchmarking GGUF variant: {level} ---")
        result = run_gguf_variant(level, gguf_path, llama_bin_dir, corpus_path, prompts[0])
        report["gguf_variants"].append(result)
        save_report()
        print(result)

    print(f"\nSaved benchmark report to {report_path}")
    print_summary_table(report["hf_variants"], report["gguf_variants"])


if __name__ == "__main__":
    main()
