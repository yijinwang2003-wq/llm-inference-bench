# LLM Inference Bench

`llm-inference-bench` is an ML systems benchmark project for studying LLM inference performance across precision modes and batch sizes. The current experiment evaluates `meta-llama/Llama-3.2-3B-Instruct` with HuggingFace Transformers on a Colab CUDA GPU using FP16, INT8, and INT4.

The main goal is to measure the practical tradeoffs between throughput, latency, and allocated GPU memory before adding production-oriented serving features such as TTFT and concurrency.

## Motivation

Batching and quantization are common inference optimization tools, but they optimize different constraints. Batching often improves aggregate throughput by increasing GPU utilization. Quantization can reduce memory pressure and make larger models or batches feasible, but it does not automatically improve serving throughput.

This project keeps the experiment controlled: fixed prompts, deterministic generation, consistent run settings, raw CSV outputs, and reproducible plots.

## Experiment Setup

| Setting | Value |
| --- | --- |
| Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Backends | HuggingFace Transformers, vLLM |
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
|   |-- precision_memory_vs_batch.png
|   |-- backend_aggregated_metrics.csv
|   |-- backend_throughput_vs_batch.png
|   |-- backend_latency_vs_batch.png
|   `-- backend_memory_vs_batch.png
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

Run the same plotting command after either experiment:

```bash id="4x9xpw"
python scripts/plot_results.py
```

The script automatically detects which CSVs are present. If only precision CSVs exist, it writes precision comparison plots. If vLLM CSVs are also present, it writes both precision and backend comparison plots.

## Results

### Precision Matrix

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

### Backend Comparison

Backend comparison artifacts:

- `outputs/backend_aggregated_metrics.csv`
- `outputs/backend_throughput_vs_batch.png`
- `outputs/backend_latency_vs_batch.png`
- `outputs/backend_memory_vs_batch.png`

| Backend | Batch Size | Mean Latency | P95 Latency | Throughput | Max GPU Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| HuggingFace FP16 | 1 | 1.1793 s | 1.4457 s | 27.3510 tok/s | 6.0047 GB |
| HuggingFace FP16 | 8 | 1.3215 s | 1.5181 s | 184.9648 tok/s | 6.0758 GB |
| HuggingFace FP16 | 32 | 2.2097 s | 2.2288 s | 434.4513 tok/s | 6.2963 GB |
| vLLM FP16 | 1 | 1.7032 s | 2.6746 s | 21.3900 tok/s | 11.8872 GB |
| vLLM FP16 | 8 | 1.2255 s | 1.2740 s | 198.9600 tok/s | 11.9024 GB |
| vLLM FP16 | 32 | 1.8425 s | 1.8843 s | 521.1701 tok/s | 11.9527 GB |

![Throughput vs Batch Size by Backend](outputs/backend_throughput_vs_batch.png)

![Latency vs Batch Size by Backend](outputs/backend_latency_vs_batch.png)

![Memory vs Batch Size by Backend](outputs/backend_memory_vs_batch.png)

## Key Findings

- FP16 was the best speed configuration in this backend, delivering the highest throughput at every batch size.
- INT8 and INT4 traded speed for memory: both reduced allocated GPU memory substantially, but neither improved throughput.
- INT4 was the strongest memory-saving mode, reaching the lowest allocated memory across all batch sizes.
- Batching improved throughput for every precision, showing that batch size remains a powerful utilization lever even when quantization slows generation.
- Batch size 32 produced the highest throughput for all precisions, while also increasing latency; p95 remained close to mean, indicating slower but stable batch execution.
- vLLM FP16 outperformed HuggingFace FP16 at batch sizes 8 and 32, reaching 521.1701 tok/s at batch size 32 versus 434.4513 tok/s for HuggingFace.
- HuggingFace FP16 was faster at batch size 1, while vLLM scaled better as batch size increased.
- vLLM used substantially more allocated GPU memory in this run, about 11.95 GB at batch size 32 versus about 6.30 GB for HuggingFace FP16.

In this HuggingFace + bitsandbytes setup, quantization is a memory optimization, not a throughput optimization. FP16 dominates on speed; INT4 dominates on memory efficiency.

For backend comparison, vLLM's serving-oriented execution model improves throughput at larger batch sizes, consistent with its PagedAttention and continuous batching design goals. The tradeoff in this experiment is higher allocated GPU memory.

## Limitations and Next Steps

- This is a HuggingFace Transformers and bitsandbytes benchmark, not a universal quantization result.
- Backend implementation matters; results may differ with other kernels, GPUs, longer outputs, or model sizes.
- Colab GPU hardware can vary across sessions.
- The benchmark uses a fixed 30-prompt set and `max_new_tokens=32`.
- The benchmark reports full generation latency, not time to first token.
- Future work should measure TTFT, add concurrent request and sustained-load tests, split prefill/decode timing, and evaluate longer outputs.
