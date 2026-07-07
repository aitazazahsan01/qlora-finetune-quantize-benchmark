# LoRA/QLoRA Fine-Tuning + Quantization Pipeline

Fine-tune `Qwen/Qwen2.5-3B-Instruct` with QLoRA on an Alpaca-style instruction
dataset, then quantize the result (bitsandbytes + GGUF) and benchmark speed
vs. quality across quantization levels. Built to run on a free Google Colab
T4 GPU.

## Stages

| # | Script | What it does |
|---|--------|---------------|
| 1 | `src/data.py` | Load, clean, format `yahma/alpaca-cleaned`, split train/val |
| 2 | `src/train.py` | QLoRA fine-tune (4-bit base + LoRA adapters) |
| 3 | `src/evaluate.py` | Perplexity + qualitative comparison, base vs fine-tuned |
| 4 | `src/quantize.py` | Merge adapters, export bitsandbytes + GGUF variants |
| 5 | `src/benchmark.py` | Speed/memory/quality benchmark across all variants |

`notebooks/colab_runner.ipynb` is the Colab entry point: installs deps,
mounts Google Drive (for checkpoint persistence across session
disconnects), and calls into `src/*.py`.

## Local setup (CPU-only sanity checks)

Stage 1 (`src/data.py`) can be run locally without a GPU to sanity-check
the data pipeline:

```bash
pip install -r requirements.txt
python src/data.py
```

Everything from stage 2 onward needs a CUDA GPU — run those via
`notebooks/colab_runner.ipynb` on Colab.

## Config

All hyperparameters live in `configs/qlora_3b.yaml`, with comments
explaining why each value was chosen for T4's constraints.
