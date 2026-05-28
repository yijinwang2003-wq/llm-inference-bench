"""Run the precision x batch-size benchmark matrix."""

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
    add_precision,
    is_oom_error,
    load_prompts,
    output_path_for_batch,
    print_summary,
)
from src.benchmark import run_benchmark
from src.config import MODEL_NAME
from src.model_loader import load_model
from src.results import save_rows_to_csv


FULL_PRECISIONS = ["fp16", "int8", "int4"]
DRY_RUN_PRECISIONS = ["fp16"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full precision x batch-size benchmark matrix."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run a quick script validation using TinyLlama, fp16 only, 3 prompts, "
            "1 timed run, no warmup, and 8 max new tokens."
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


def release_model(model) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    model_name = args.model or (DRY_RUN_MODEL if args.dry_run else MODEL_NAME)
    precisions = DRY_RUN_PRECISIONS if args.dry_run else FULL_PRECISIONS

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

    print("Quantization benchmark matrix", flush=True)
    print(f"  model: {model_name}", flush=True)
    print(f"  precisions: {precisions}", flush=True)
    print(f"  batch sizes: {BATCH_SIZES}", flush=True)
    print(f"  prompts: {len(prompts)}", flush=True)
    print(f"  num_runs: {num_runs}", flush=True)
    print(f"  warmup_runs: {warmup_runs}", flush=True)
    print(f"  max_new_tokens: {max_new_tokens}", flush=True)
    if args.dry_run:
        print("  mode: dry run", flush=True)

    completed = []
    failed = []
    for precision in precisions:
        print(f"\nLoading {precision} model once for all batch sizes...", flush=True)
        try:
            model, tokenizer, device = load_model(model_name, precision=precision)
        except RuntimeError as exc:
            failed.append((precision, "load"))
            print(
                f"\nFailed to load {precision} model: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue

        print(f"Loaded {precision} model on device: {device}", flush=True)
        try:
            for batch_size in BATCH_SIZES:
                output_path = output_path_for_batch(
                    batch_size,
                    precision=precision,
                    dry_run=args.dry_run,
                )
                print(f"\nRunning {precision} batch size {batch_size}...", flush=True)
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
                    rows = add_precision(rows, precision)
                    save_rows_to_csv(rows, output_path)
                    print_summary(precision, batch_size, rows, output_path)
                    completed.append((precision, batch_size))
                except RuntimeError as exc:
                    failed.append((precision, batch_size))
                    if is_oom_error(exc):
                        print(
                            f"\n{precision} batch size {batch_size} failed with OOM. "
                            "Previous CSV outputs were left untouched.",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    print(
                        f"\n{precision} batch size {batch_size} failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
        finally:
            release_model(model)

    print("\nQuantization matrix complete", flush=True)
    print(f"  completed configs: {completed}", flush=True)
    if failed:
        print(f"  failed configs: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
