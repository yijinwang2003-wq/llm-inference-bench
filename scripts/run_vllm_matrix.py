"""Run the vLLM FP16 batch-scaling benchmark matrix."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment_matrix import (
    BATCH_SIZES,
    DRY_RUN_MAX_NEW_TOKENS,
    DRY_RUN_MODEL,
    DRY_RUN_NUM_RUNS,
    DRY_RUN_PROMPTS,
    DRY_RUN_WARMUP_RUNS,
    FULL_MAX_NEW_TOKENS,
    FULL_NUM_RUNS,
    FULL_WARMUP_RUNS,
    is_oom_error,
    load_prompts,
)
from src.config import MODEL_NAME
from src.metrics import summarize_results
from src.results import save_rows_to_csv
from src.vllm_benchmark import load_vllm_model, run_vllm_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the vLLM FP16 benchmark matrix.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run a quick vLLM validation using TinyLlama, 3 prompts, 1 timed run, "
            "no warmup, and 8 max new tokens."
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            "Override the model. Defaults to Llama for the full experiment and "
            "TinyLlama for --dry-run."
        ),
    )
    return parser.parse_args()


def output_path_for_batch(batch_size: int, dry_run: bool) -> Path:
    prefix = "dry_run_vllm_fp16" if dry_run else "vllm_fp16"
    return Path("outputs") / f"{prefix}_batch{batch_size}.csv"


def release_model(llm) -> None:
    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def print_summary(batch_size: int, rows: list[dict], output_path: Path) -> None:
    summary = summarize_results(rows)
    max_memory_allocated = max(
        (float(row.get("gpu_memory_allocated_gb", 0.0)) for row in rows),
        default=0.0,
    )
    max_memory_reserved = max(
        (float(row.get("gpu_memory_reserved_gb", 0.0)) for row in rows),
        default=0.0,
    )

    print(f"\nFinished vLLM FP16 batch size {batch_size}", flush=True)
    print(f"  rows: {len(rows)}", flush=True)
    print(f"  output: {output_path}", flush=True)
    print(
        f"  mean latency: {summary['mean_latency_seconds']:.4f} seconds",
        flush=True,
    )
    print(
        f"  p95 latency: {summary['p95_latency_seconds']:.4f} seconds",
        flush=True,
    )
    print(
        "  mean throughput: "
        f"{summary['mean_throughput_tokens_per_sec']:.4f} tokens/sec",
        flush=True,
    )
    print(f"  max GPU memory allocated: {max_memory_allocated:.4f} GB", flush=True)
    print(f"  max GPU memory reserved: {max_memory_reserved:.4f} GB", flush=True)


def main() -> int:
    args = parse_args()
    model_name = args.model or (DRY_RUN_MODEL if args.dry_run else MODEL_NAME)

    prompts = load_prompts(Path("data/prompts.json"))
    if args.dry_run:
        prompts = prompts[:DRY_RUN_PROMPTS]
        num_runs = DRY_RUN_NUM_RUNS
        warmup_runs = DRY_RUN_WARMUP_RUNS
        max_new_tokens = DRY_RUN_MAX_NEW_TOKENS
    else:
        num_runs = FULL_NUM_RUNS
        warmup_runs = FULL_WARMUP_RUNS
        max_new_tokens = FULL_MAX_NEW_TOKENS

    print("vLLM FP16 benchmark matrix", flush=True)
    print(f"  model: {model_name}", flush=True)
    print("  backend: vllm", flush=True)
    print("  precision: fp16", flush=True)
    print(f"  batch sizes: {BATCH_SIZES}", flush=True)
    print(f"  prompts: {len(prompts)}", flush=True)
    print(f"  num_runs: {num_runs}", flush=True)
    print(f"  warmup_runs: {warmup_runs}", flush=True)
    print(f"  max_new_tokens: {max_new_tokens}", flush=True)
    if args.dry_run:
        print("  mode: dry run", flush=True)

    print("\nLoading vLLM model once for all batch sizes...", flush=True)
    llm = load_vllm_model(model_name)

    completed = []
    failed = []
    try:
        for batch_size in BATCH_SIZES:
            output_path = output_path_for_batch(batch_size, dry_run=args.dry_run)
            print(f"\nRunning vLLM FP16 batch size {batch_size}...", flush=True)
            try:
                rows = run_vllm_benchmark(
                    llm=llm,
                    prompts=prompts,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    warmup_runs=warmup_runs,
                    num_runs=num_runs,
                )
                save_rows_to_csv(rows, output_path)
                print_summary(batch_size, rows, output_path)
                completed.append(batch_size)
            except RuntimeError as exc:
                failed.append(batch_size)
                if is_oom_error(exc):
                    print(
                        f"\nvLLM batch size {batch_size} failed with OOM. "
                        "Previous CSV outputs were left untouched.",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                raise
    finally:
        release_model(llm)

    print("\nvLLM matrix complete", flush=True)
    print(f"  completed batch sizes: {completed}", flush=True)
    if failed:
        print(f"  failed batch sizes: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
