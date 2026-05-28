# LLM Inference Bench

`llm-inference-bench` is a small benchmarking project for measuring LLM inference performance with HuggingFace Transformers. The current baseline evaluates FP16 generation for `meta-llama/Llama-3.2-3B-Instruct` on a CUDA GPU and compares batch size 1 against batch size 8.

The goal is to build a clear, reproducible systems performance baseline before adding more serving backends, quantization modes, and latency breakdowns.

## Motivation

LLM serving performance depends heavily on batching, model precision, hardware, backend runtime, and measurement methodology. This project focuses on one question first:

How does increasing batch size affect latency and throughput for a small FP16 Transformer inference workload?

The initial result shows that batch size 8 delivers much higher aggregate throughput while mean latency remains roughly stable for this prompt set and generation length.

## Features

- Fixed prompt generation for repeatable benchmark inputs.
- HuggingFace Transformers FP16 inference path.
- CUDA execution when a GPU is available, with CPU fallback for smoke tests.
- Configurable model, batch size, run count, warmup count, output length, and prompt limit.
- CSV output with per-prompt latency, throughput, token counts, and GPU memory fields.
- Plotting pipeline using pandas and matplotlib.
- Publication-style throughput, mean latency, and p95 latency plots.

## Repo Structure

```text
llm-inference-bench/
|-- data/
|   `-- prompts.json
|-- outputs/
|   |-- fp16_batch1.csv
|   |-- fp16_batch8.csv
|   |-- fp16_batch32.csv
|   |-- fp16_aggregated_metrics.csv
|   |-- fp16_throughput_vs_batch.png
|   |-- fp16_mean_latency_vs_batch.png
|   |-- fp16_p95_latency_vs_batch.png
|   `-- fp16_memory_vs_batch.png
|-- scripts/
|   |-- generate_prompts.py
|   |-- run_experiment_matrix.py
|   |-- plot_results.py
|   `-- run_benchmark.py
|-- src/
|   |-- benchmark.py
|   |-- config.py
|   |-- metrics.py
|   |-- model_loader.py
|   `-- results.py
|-- README.md
|-- WRITEUP.md
`-- requirements.txt
```

## Setup

Use Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Generate the fixed 30-prompt set:

```bash
python scripts/generate_prompts.py
```

This writes `data/prompts.json`.

## HuggingFace Login

The Llama model is gated on HuggingFace. Before running `meta-llama/Llama-3.2-3B-Instruct`, accept the model license on HuggingFace and authenticate in your runtime:

```bash
huggingface-cli login
```

For Colab, run the login command in a notebook cell or provide a HuggingFace token through your preferred Colab secrets workflow.

## Colab GPU Instructions

1. Open the project in Colab or clone it into a Colab notebook runtime.
2. Select a GPU runtime: `Runtime` > `Change runtime type` > `T4`, `L4`, `A100`, or another CUDA GPU.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Log in to HuggingFace:

```bash
huggingface-cli login
```

5. Generate prompts if needed:

```bash
python scripts/generate_prompts.py
```

6. Run the smoke test and benchmark commands below.

## TinyLlama Smoke Test

Use this command to validate the pipeline without downloading the Llama 3.2 3B model:

```bash
python scripts/run_benchmark.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --batch-size 1 \
  --num-runs 1 \
  --warmup-runs 0 \
  --max-new-tokens 8 \
  --limit-prompts 1 \
  --output outputs/tinyllama_smoke.csv
```

On Colab GPU, you can run a slightly broader smoke test:

```bash
python scripts/run_benchmark.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --batch-size 1 \
  --num-runs 1 \
  --warmup-runs 0 \
  --max-new-tokens 8 \
  --output outputs/tinyllama_colab_smoke.csv
```

## FP16 Matrix Dry Run

Use the matrix dry run to test the end-to-end experiment script quickly. This uses 3 prompts, 1 timed run, no warmup, `max_new_tokens=8`, and TinyLlama by default:

```bash
python scripts/run_experiment_matrix.py --dry-run
```

Dry-run outputs are written as `outputs/dry_run_fp16_batch1.csv`, `outputs/dry_run_fp16_batch8.csv`, and `outputs/dry_run_fp16_batch32.csv` so they do not overwrite full experiment results.

To dry-run the same script with a specific model, pass `--model`:

```bash
python scripts/run_experiment_matrix.py \
  --dry-run \
  --model meta-llama/Llama-3.2-3B-Instruct
```

## Run Llama FP16 Benchmarks

For the complete FP16 batch-scaling experiment on Colab GPU, use the matrix runner:

```bash
python scripts/run_experiment_matrix.py
```

This loads `meta-llama/Llama-3.2-3B-Instruct` once and runs batch sizes `1`, `8`, and `32` in increasing order with:

- `num_runs=5`
- `warmup_runs=1`
- `max_new_tokens=32`
- all 30 prompts

It writes each CSV immediately after the corresponding batch size finishes:

- `outputs/fp16_batch1.csv`
- `outputs/fp16_batch8.csv`
- `outputs/fp16_batch32.csv`

Batch size 32 may run out of memory on smaller Colab GPUs. If that happens, previously completed CSV files are left in place.

You can still run individual benchmark commands manually. The earlier small batch comparison used `meta-llama/Llama-3.2-3B-Instruct`, `max_new_tokens=32`, `num_runs=3`, and the first 8 prompts.

Batch size 1:

```bash
python scripts/run_benchmark.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --batch-size 1 \
  --num-runs 3 \
  --warmup-runs 2 \
  --max-new-tokens 32 \
  --limit-prompts 8 \
  --output outputs/benchmark_results_b1.csv
```

Batch size 8:

```bash
python scripts/run_benchmark.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --batch-size 8 \
  --num-runs 3 \
  --warmup-runs 2 \
  --max-new-tokens 32 \
  --limit-prompts 8 \
  --output outputs/benchmark_results_b8.csv
```

Supported batch sizes are `1`, `8`, and `32`.

## Generate Plots

After producing the FP16 matrix CSVs, generate the aggregate CSV and plots:

```bash
python scripts/plot_results.py
```

For the new FP16 matrix outputs, this writes:

- `outputs/fp16_aggregated_metrics.csv`
- `outputs/fp16_throughput_vs_batch.png`
- `outputs/fp16_mean_latency_vs_batch.png`
- `outputs/fp16_p95_latency_vs_batch.png`
- `outputs/fp16_memory_vs_batch.png`

The plotter still falls back to the older `outputs/benchmark_results_b1.csv` and `outputs/benchmark_results_b8.csv` files when no `outputs/fp16_batch*.csv` files are present.

## Output Files

Raw benchmark CSVs contain per-prompt measurements:

- `run_index`: timed run index.
- `batch_index`: batch index within the run.
- `batch_size`: number of prompts processed together.
- `prompt_index`: prompt index from `data/prompts.json`.
- `prompt`: input text.
- `latency_seconds`: measured generation latency for the batch.
- `generated_tokens`: generated token count used for throughput.
- `throughput_tokens_per_sec`: generated tokens per second.
- `gpu_memory_allocated_gb`: CUDA allocated memory after generation.
- `gpu_memory_reserved_gb`: CUDA reserved memory after generation.
- `prompt_length_tokens`: tokenized prompt length.
- `output_length_tokens`: prompt plus generated output length.

`outputs/fp16_aggregated_metrics.csv` summarizes:

- `batch_size`
- `mean_latency`
- `p95_latency`
- `throughput`
- `max_gpu_memory_allocated_gb`
- `max_gpu_memory_reserved_gb`

The PNG files visualize throughput, mean latency, p95 latency, and GPU memory across batch sizes.

## Key Result

For this early FP16 baseline on Colab GPU:

| Batch Size | Mean Latency | P95 Latency | Throughput |
| ---: | ---: | ---: | ---: |
| 1 | 1.2773 s | 1.5802 s | 25.3276 tokens/sec |
| 8 | 1.3037 s | 1.4364 s | 197.3477 tokens/sec |

Increasing batch size from 1 to 8 improved throughput by almost 8x while mean latency stayed roughly stable around 1.3 seconds.

This is an early FP16 baseline, not a complete production serving benchmark.

## Next Steps

- Add INT8 benchmarking.
- Add INT4 benchmarking.
- Add vLLM backend support.
- Measure time to first token (TTFT).
- Compare more GPUs and longer generation lengths.
- Add richer serving metrics such as queueing delay and concurrency behavior.
