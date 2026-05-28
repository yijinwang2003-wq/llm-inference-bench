"""CLI entrypoint for the FP16 Transformers benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_prompts import main as generate_prompts
from src.config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_RUNS,
    DEFAULT_WARMUP_RUNS,
    MODEL_NAME,
    SUPPORTED_BATCH_SIZES,
)
from src.metrics import summarize_results
from src.results import save_rows_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FP16 Transformers benchmark.")
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=SUPPORTED_BATCH_SIZES,
        default=1,
        help="Number of prompts to process per generation batch.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generated tokens per prompt.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
        help="Number of warmup generations before timed runs.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help="Number of timed benchmark passes over the prompt set.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="HuggingFace model ID to benchmark.",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        help="Use only the first N prompts from the prompt file.",
    )
    parser.add_argument(
        "--output",
        default="outputs/benchmark_results.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    if not path.exists():
        print(f"{path} not found. Generating default prompts first.")
        generate_prompts()

    with path.open("r", encoding="utf-8") as file:
        prompt_items = json.load(file)

    prompts = [item["prompt"] for item in prompt_items]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _torch_version_tuple() -> tuple[int, ...]:
    raw_version = version("torch").split("+", maxsplit=1)[0]
    parts = []
    for part in raw_version.split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts)


def validate_environment() -> None:
    if _torch_version_tuple() < (2, 4):
        raise RuntimeError(
            f"Installed torch version is {version('torch')}, but this benchmark "
            "requires torch >= 2.4 for the installed Transformers package. Run:\n\n"
            "pip install -r requirements.txt"
        )


def main() -> int:
    args = parse_args()
    prompt_path = Path("data/prompts.json")

    try:
        prompts = load_prompts(prompt_path)
        if args.limit_prompts is not None:
            if args.limit_prompts < 1:
                raise ValueError("--limit-prompts must be positive")
            prompts = prompts[: args.limit_prompts]
        validate_environment()

        from src.benchmark import run_benchmark
        from src.model_loader import load_model_fp16

        print(f"Loaded {len(prompts)} prompts from {prompt_path}", flush=True)
        print(f"Loading model: {args.model}", flush=True)
        model, tokenizer, device = load_model_fp16(args.model)
        print(f"Running benchmark on device: {device}", flush=True)

        rows = run_benchmark(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            warmup_runs=args.warmup_runs,
            num_runs=args.num_runs,
        )
        save_rows_to_csv(rows, args.output)

        summary = summarize_results(rows)
        print(f"Saved {len(rows)} rows to {args.output}", flush=True)
        print("Summary:", flush=True)
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}", flush=True)
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
