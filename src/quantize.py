"""Stage 4: merge LoRA adapter into base weights, then quantize.

Two very different kinds of "quantization" show up across this pipeline:

- bitsandbytes (used in train.py/evaluate.py) is *load-time* quantization -
  it quantizes fp16 weights to 4/8-bit the moment they're loaded into VRAM,
  transparently to the rest of the code. There's no separate "bnb file" to
  export here; benchmark.py just loads this same merged fp16 checkpoint with
  different bnb configs and measures the result at load time.
- GGUF is a *static* pre-quantized file format used by llama.cpp - the
  quantization happens once, offline, producing a standalone file loaded
  directly at whatever bit-width it was baked at. That's what this script
  produces on disk.

Needs a CUDA GPU (for the merge step) - run this on Colab. Also needs git +
a C/C++ toolchain + cmake to build llama.cpp, all present by default on
Colab runtimes.
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import load_config, resolve_path  # noqa: E402

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def run(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def merge_adapter(cfg):
    model_cfg = cfg["model"]
    adapter_dir = str(resolve_path(cfg["training"]["output_dir"]))
    merged_dir = resolve_path(cfg["quantization"]["merged_dir"])

    if merged_dir.exists() and any(merged_dir.iterdir()):
        print(f"Merged model already exists at {merged_dir}, skipping merge.")
        return merged_dir

    # Merge against an *unquantized* fp16 base, not the 4-bit training copy -
    # baking a LoRA delta into already-quantized weights compounds
    # quantization error unnecessarily when a clean fp16 copy is one
    # download away.
    print("Loading base model in fp16 (unquantized) for a clean merge ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_id"], torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_id"])

    print(f"Attaching adapter from {adapter_dir} and merging ...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged = peft_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Saved merged fp16 model to {merged_dir}")
    return merged_dir


def ensure_llama_cpp(cfg):
    llama_dir = resolve_path(cfg["quantization"]["llama_cpp_dir"])
    bin_dir = llama_dir / "build" / "bin"
    quantize_bin = bin_dir / "llama-quantize"
    convert_script = llama_dir / "convert_hf_to_gguf.py"

    if not llama_dir.exists():
        llama_dir.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/ggerganov/llama.cpp",
                str(llama_dir),
            ]
        )

    # llama-quantize is used here (stage 4); llama-cli, llama-perplexity and
    # llama-bench are used by benchmark.py (stage 5).
    targets = ["llama-quantize", "llama-cli", "llama-perplexity", "llama-bench"]
    if not all((bin_dir / t).exists() for t in targets):
        print(f"Building llama.cpp ({', '.join(targets)}) ...")
        run(["cmake", "-B", "build"], cwd=str(llama_dir))
        # Cap parallel compile jobs and build only the targets we need.
        # Unbounded `-j` spawns one compiler process per core, and building
        # the default `all` target also compiles every CLI llama.cpp ships
        # (llava, minicpmv, gemma3, qwen2vl-cli, ...) that this pipeline never
        # uses. On Colab's ~12GB-RAM runtimes that combination reliably OOMs
        # and kills the runtime mid-build.
        jobs = max(1, min(4, os.cpu_count() or 2))
        run(
            [
                "cmake",
                "--build",
                "build",
                "--config",
                "Release",
                *sum((["--target", t] for t in targets), []),
                "-j",
                str(jobs),
            ],
            cwd=str(llama_dir),
        )

    if not convert_script.exists():
        raise FileNotFoundError(
            f"Expected {convert_script} - llama.cpp's repo layout may have "
            "changed; check https://github.com/ggerganov/llama.cpp for the "
            "current HF->GGUF conversion script name."
        )

    return llama_dir, quantize_bin, convert_script


def convert_and_quantize(cfg, merged_dir, quantize_bin, convert_script):
    gguf_dir = resolve_path(cfg["quantization"]["gguf_dir"])
    gguf_dir.mkdir(parents=True, exist_ok=True)

    f16_gguf = gguf_dir / "model-f16.gguf"
    if not f16_gguf.exists():
        print("Converting merged HF model -> f16 GGUF ...")
        run(
            [
                sys.executable,
                str(convert_script),
                str(merged_dir),
                "--outfile",
                str(f16_gguf),
                "--outtype",
                "f16",
            ]
        )
    else:
        print(f"{f16_gguf} already exists, skipping conversion.")

    for level in cfg["quantization"]["gguf_levels"]:
        out_path = gguf_dir / f"model-{level}.gguf"
        if out_path.exists():
            print(f"{out_path} already exists, skipping.")
            continue
        print(f"Quantizing to {level} ...")
        run([str(quantize_bin), str(f16_gguf), str(out_path), level])

    return f16_gguf


def main():
    cfg = load_config()
    merged_dir = merge_adapter(cfg)
    _, quantize_bin, convert_script = ensure_llama_cpp(cfg)
    f16_gguf = convert_and_quantize(cfg, merged_dir, quantize_bin, convert_script)

    gguf_dir = resolve_path(cfg["quantization"]["gguf_dir"])
    print("\nDone. Artifacts:")
    print(f"  Merged fp16 HF model: {merged_dir}")
    print(f"  GGUF f16:             {f16_gguf}")
    for level in cfg["quantization"]["gguf_levels"]:
        print(f"  GGUF {level}: {gguf_dir / f'model-{level}.gguf'}")


if __name__ == "__main__":
    main()
