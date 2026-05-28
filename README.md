# LLM Inference Bench

`llm-inference-bench` is an ML systems benchmark project for studying LLM inference performance across precision modes and batch sizes. The current experiment evaluates `meta-llama/Llama-3.2-3B-Instruct` with HuggingFace Transformers on a Colab CUDA GPU using FP16, INT8, and INT4.

The main goal is to measure the practical tradeoffs between throughput, latency, and allocated GPU memory before moving to production-oriented serving features such as TTFT, vLLM, concurrency, and prefill/decode breakdowns.

## Motivation

Batching and quantization are common inference optimization tools, but they optimize different constraints. Batching often improves aggregate throughput by increasing GPU utilization. Quantization can reduce memory pressure and make larger models or batches feasible, but it does not automatically improve serving throughput.

This project keeps the experiment controlled: fixed prompts, deterministic generation, consistent run settings, raw CSV outputs, and reproducible plots.

## Experiment Setup

| Setting | Value |
| --- | --- |
| Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Backend | HuggingFace Transformers |
| Quantization | bitsandbytes for INT8 and INT4 |
| Hardware | CUDA GPU on Google Colab |
| Precisions | FP16, INT8, INT4 |
| Batch sizes | 1, 8, 32 |
| Prompt set | 30 fixed prompts |
| Runs | 5 |
| Warmup runs | 1 |
| Max new tokens | 32 |

## Benchmark Methodology

The benchmark loads a model, runs deterministic generation over the fixed prompt set, and records per-batch latency, generated-token throughput, token counts, and CUDA memory fields. The quantization matrix loads the model once per precision, then runs batch sizes 1, 8, and 32 in increasing order.

Each configuration writes its CSV immediately after completion, so earlier results are preserved if a later precision or batch size fails.

## Repo Structure

```text
llm-inference-bench/
|-- data/
|   `-- prompts.json
|-- outputs/
|   |-- precision_aggregated_metrics.csv
|   |-- precision_throughput_vs_batch.png
|   |-- precision_latency_vs_batch.png
|   `-- precision_memory_vs_batch.png
|-- scripts/
|   |-- generate_prompts.py
|   |-- run_experiment_matrix.py
|   |-- run_quantization_matrix.py
|   |-- run_vllm_matrix.py
|   |-- plot_results.py
|   `-- run_benchmark.py
|-- src/
|   |-- benchmark.py
|   |-- config.py
|   |-- metrics.py
|   |-- model_loader.py
|   |-- vllm_benchmark.py
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

The Llama model is gated on HuggingFace. Accept the model license and authenticate before running the full experiment:

```bash
huggingface-cli login
```

## Dry Run

The quantization matrix includes a dry-run mode for checking the script and output paths. It uses TinyLlama, FP16 only, 3 prompts, 1 timed run, no warmup, and `max_new_tokens=8`:

```bash
python scripts/run_quantization_matrix.py --dry-run
```

Dry-run outputs are written to `outputs/dry_run_fp16_batch*.csv`.

## Full Precision Matrix on Colab GPU

Run the complete precision x batch-size experiment on a Colab GPU runtime:

```bash
python scripts/run_quantization_matrix.py
```

This evaluates:

- Precisions: `fp16`, `int8`, `int4`
- Batch sizes: `1`, `8`, `32`

Expected raw CSV outputs include:

- `outputs/fp16_batch1.csv`, `outputs/fp16_batch8.csv`, `outputs/fp16_batch32.csv`
- `outputs/int8_batch1.csv`, `outputs/int8_batch8.csv`, `outputs/int8_batch32.csv`
- `outputs/int4_batch1.csv`, `outputs/int4_batch8.csv`, `outputs/int4_batch32.csv`

## vLLM Backend Benchmark

Phase 2 adds a separate vLLM FP16 path for backend comparison against the HuggingFace Transformers FP16 baseline. vLLM is designed for serving-oriented inference and uses techniques such as PagedAttention and continuous batching to improve memory management and request scheduling.

Run the vLLM dry run on a CUDA runtime:

```bash
python scripts/run_vllm_matrix.py --dry-run
```

Run the full vLLM FP16 matrix on Colab GPU:

```bash
python scripts/run_vllm_matrix.py
```

This writes:

- `outputs/vllm_fp16_batch1.csv`
- `outputs/vllm_fp16_batch8.csv`
- `outputs/vllm_fp16_batch32.csv`

## Generate Plots

After the precision matrix CSVs exist, generate aggregate metrics and plots:

```bash
python scripts/plot_results.py
```

This writes:

- `outputs/precision_aggregated_metrics.csv`
- `outputs/precision_throughput_vs_batch.png`
- `outputs/precision_latency_vs_batch.png`
- `outputs/precision_memory_vs_batch.png`

When both Transformers FP16 and vLLM FP16 CSVs are present, the same plotting command also writes backend comparison outputs:

- `outputs/backend_aggregated_metrics.csv`
- `outputs/backend_throughput_vs_batch.png`
- `outputs/backend_latency_vs_batch.png`
- `outputs/backend_memory_vs_batch.png`

## Results

| Precision | Batch Size | Mean Latency | P95 Latency | Throughput | Max GPU Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | 1 | 1.1793 s | 1.4457 s | 27.3510 tok/s | 6.0047 GB |
| FP16 | 8 | 1.3215 s | 1.5181 s | 184.9648 tok/s | 6.0758 GB |
| FP16 | 32 | 2.2097 s | 2.2288 s | 434.4513 tok/s | 6.2963 GB |
| INT8 | 1 | 4.1999 s | 4.6472 s | 7.6495 tok/s | 3.3836 GB |
| INT8 | 8 | 5.5006 s | 5.9852 s | 44.2865 tok/s | 3.4788 GB |
| INT8 | 32 | 6.3974 s | 6.6735 s | 150.2164 tok/s | 3.7589 GB |
| INT4 | 1 | 2.0525 s | 2.4902 s | 15.7128 tok/s | 2.3250 GB |
| INT4 | 8 | 5.4925 s | 5.6064 s | 44.3022 tok/s | 2.3983 GB |
| INT4 | 32 | 6.9431 s | 6.9532 s | 138.2671 tok/s | 2.6105 GB |

![Throughput vs Batch Size by Precision](outputs/precision_throughput_vs_batch.png)

![Latency vs Batch Size by Precision](outputs/precision_latency_vs_batch.png)

![Memory vs Batch Size by Precision](outputs/precision_memory_vs_batch.png)

## Key Findings

- FP16 was the best speed configuration in this backend, delivering the highest throughput at every batch size.
- INT8 and INT4 traded speed for memory: both reduced allocated GPU memory substantially, but neither improved throughput.
- INT4 was the strongest memory-saving mode, reaching the lowest allocated memory across all batch sizes.
- Batching improved throughput for every precision, showing that batch size remains a powerful utilization lever even when quantization slows generation.
- Batch size 32 produced the highest throughput for all precisions, while also increasing latency; p95 remained close to mean, indicating slower but stable batch execution.

In this HuggingFace + bitsandbytes setup, quantization is a memory optimization, not a throughput optimization. FP16 dominates on speed; INT4 dominates on memory efficiency.

## Limitations and Next Steps

- This is a HuggingFace Transformers and bitsandbytes benchmark, not a universal quantization result.
- Backend implementation matters; results may differ with vLLM, other kernels, other GPUs, longer outputs, or different model sizes.
- Colab GPU hardware can vary across sessions.
- The benchmark uses a fixed 30-prompt set and `max_new_tokens=32`.
- The benchmark reports full generation latency, not time to first token.
- Future work should measure TTFT, compare vLLM, add concurrent request and sustained-load tests, split prefill/decode timing, and evaluate longer outputs.
