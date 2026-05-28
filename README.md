# LLM Inference Bench

A minimal benchmark pipeline for measuring HuggingFace Transformers inference performance. The MVP focuses on a single FP16 baseline for `meta-llama/Llama-3.2-3B-Instruct`; vLLM and quantized variants are intentionally left for later iterations.

## Setup

Use Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Llama models on HuggingFace may require accepting the model license and authenticating:

```bash
huggingface-cli login
```

## Generate Prompts

The benchmark uses a fixed prompt set with 5 short, 5 medium, and 5 long prompts.

```bash
python scripts/generate_prompts.py
```

This writes `data/prompts.json`.

## Run FP16 Benchmark

Run the default batch size 1 benchmark:

```bash
python scripts/run_benchmark.py
```

Override batch size and run counts:

```bash
python scripts/run_benchmark.py --batch-size 8 --num-runs 3 --warmup-runs 2
python scripts/run_benchmark.py --batch-size 32 --num-runs 3 --warmup-runs 2
```

Override the model:

```bash
python scripts/run_benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
python scripts/run_benchmark.py --model meta-llama/Llama-3.2-3B-Instruct
```

Supported batch sizes are `1`, `8`, and `32`. CUDA is used automatically when available; otherwise the script falls back to CPU with float32.

Fastest local CPU smoke test:

```bash
python scripts/run_benchmark.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --batch-size 1 \
  --num-runs 1 \
  --warmup-runs 0 \
  --max-new-tokens 8 \
  --limit-prompts 1
```

Safer Colab smoke test:

```bash
python scripts/run_benchmark.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --batch-size 1 \
  --num-runs 1 \
  --warmup-runs 0 \
  --max-new-tokens 8
```

Colab GPU Llama smoke test:

```bash
python scripts/run_benchmark.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --batch-size 1 \
  --num-runs 1 \
  --warmup-runs 0 \
  --max-new-tokens 8
```

## Output CSV

By default, results are saved to `outputs/benchmark_results.csv`.

Expected columns:

- `run_index`
- `batch_index`
- `batch_size`
- `prompt_index`
- `prompt`
- `latency_seconds`
- `generated_tokens`
- `throughput_tokens_per_sec`
- `gpu_memory_allocated_gb`
- `gpu_memory_reserved_gb`
- `prompt_length_tokens`
- `output_length_tokens`

## Next Steps

- Add INT8 benchmarking.
- Add INT4 benchmarking.
- Add vLLM backend support.
- Add time-to-first-token (TTFT) measurement.
