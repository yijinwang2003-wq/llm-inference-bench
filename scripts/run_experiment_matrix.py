"""Run a single-precision batch-scaling benchmark matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_prompts import main as generate_prompts
from src.benchmark import run_benchmark
from src.config import MODEL_NAME
from src.metrics import summarize_results
from src.model_loader import SUPPORTED_PRECISIONS, load_model
from src.results import save_rows_to_csv


BATCH_SIZES = [1, 8, 32]
FULL_NUM_RUNS = 5
FULL_WARMUP_RUNS = 1
FULL_MAX_NEW_TOKENS = 32

DRY_RUN_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DRY_RUN_PROMPTS = 3
DRY_RUN_NUM_RUNS = 1
DRY_RUN_WARMUP_RUNS = 0
DRY_RUN_MAX_NEW_TOKENS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single-precision batch-scaling experiment matrix."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run a quick script validation using 3 prompts, 1 timed run, no warmup, "
            "8 max new tokens, and TinyLlama unless --model is provided."
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            "Override the model. Defaults to Llama for the full experiment and "
            "TinyLlama for --dry-run."
        ),
    )
    parser.add_argument(
        "--precision",
        choices=SUPPORTED_PRECISIONS,
        default="fp16",
        help="Model precision to benchmark.",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    if not path.exists():
        print(f"{path} not found. Generating prompt set first.", flush=True)
        generate_prompts()

    with path.open("r", encoding="utf-8") as file:
        prompt_items = json.load(file)

    prompts = [item["prompt"] for item in prompt_items]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def output_path_for_batch(batch_size: int, precision: str, dry_run: bool) -> Path:
    prefix = f"dry_run_{precision}" if dry_run else precision
    return Path("outputs") / f"{prefix}_batch{batch_size}.csv"


def is_oom_error(error: Exception) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def add_precision(rows: list[dict], precision: str) -> list[dict]:
    return [{**row, "precision": precision} for row in rows]


def print_summary(
    precision: str,
    batch_size: int,
    rows: list[dict],
    output_path: Path,
) -> None:
    summary = summarize_results(rows)
    max_memory_allocated = max(
        (float(row.get("gpu_memory_allocated_gb", 0.0)) for row in rows),
        default=0.0,
    )
    max_memory_reserved = max(
        (float(row.get("gpu_memory_reserved_gb", 0.0)) for row in rows),
        default=0.0,
    )

    print(f"\nFinished {precision} batch size {batch_size}", flush=True)
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
    prompt_path = Path("data/prompts.json")
    model_name = args.model or (DRY_RUN_MODEL if args.dry_run else MODEL_NAME)

    prompts = load_prompts(prompt_path)
    if args.dry_run:
        prompts = prompts[:DRY_RUN_PROMPTS]
        num_runs = DRY_RUN_NUM_RUNS
        warmup_runs = DRY_RUN_WARMUP_RUNS
        max_new_tokens = DRY_RUN_MAX_NEW_TOKENS
    else:
        num_runs = FULL_NUM_RUNS
        warmup_runs = FULL_WARMUP_RUNS
        max_new_tokens = FULL_MAX_NEW_TOKENS

    print("Batch-scaling experiment", flush=True)
    print(f"  model: {model_name}", flush=True)
    print(f"  precision: {args.precision}", flush=True)
    print(f"  batch sizes: {BATCH_SIZES}", flush=True)
    print(f"  prompts: {len(prompts)}", flush=True)
    print(f"  num_runs: {num_runs}", flush=True)
    print(f"  warmup_runs: {warmup_runs}", flush=True)
    print(f"  max_new_tokens: {max_new_tokens}", flush=True)
    if args.dry_run:
        print("  mode: dry run", flush=True)

    print("\nLoading model once for all batch sizes...", flush=True)
    model, tokenizer, device = load_model(model_name, precision=args.precision)
    print(f"Loaded model on device: {device}", flush=True)

    completed = []
    failed = []
    for batch_size in BATCH_SIZES:
        output_path = output_path_for_batch(
            batch_size,
            precision=args.precision,
            dry_run=args.dry_run,
        )
        print(f"\nRunning {args.precision} batch size {batch_size}...", flush=True)
        try:
            rows = run_benchmark(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                warmup_runs=warmup_runs,
                num_runs=num_runs,
            )
            rows = add_precision(rows, args.precision)
            save_rows_to_csv(rows, output_path)
            print_summary(args.precision, batch_size, rows, output_path)
            completed.append(batch_size)
        except RuntimeError as exc:
            if is_oom_error(exc):
                failed.append(batch_size)
                print(
                    f"\nBatch size {batch_size} failed with OOM. "
                    "Previous CSV outputs were left untouched.",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            raise

    print("\nExperiment matrix complete", flush=True)
    print(f"  completed batch sizes: {completed}", flush=True)
    if failed:
        print(f"  failed batch sizes: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
