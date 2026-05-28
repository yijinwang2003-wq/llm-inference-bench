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

- FP16 had the highest throughput in this HuggingFace Transformers setup.
- INT8 and INT4 significantly reduced allocated GPU memory.
- Quantization did not improve throughput here; generation was slower with bitsandbytes INT8 and INT4.
- Batch size increased throughput for every precision mode.
- Batch size 32 had the highest throughput for every precision, with higher latency.
- INT4 used the least allocated GPU memory.

## Limitations

- This is a HuggingFace Transformers and bitsandbytes benchmark, not a universal quantization result.
- Quantization can reduce memory without guaranteeing throughput gains; backend implementation matters.
- Colab GPU hardware can vary across sessions.
- The benchmark uses a fixed 30-prompt set and `max_new_tokens=32`.
- The benchmark reports full generation latency, not time to first token.
- It does not model multi-user traffic, request queueing, streaming, or production service-level objectives.

## Future Work

- Measure time to first token (TTFT).
- Add vLLM backend comparisons.
- Add concurrent request and sustained-load benchmarks.
- Separate prefill and decode timing.
- Test longer output lengths and larger prompt sets.
- Compare additional GPU types.
