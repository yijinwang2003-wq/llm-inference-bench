"""Generate publication-style benchmark plots from CSV outputs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs")
PLOT_CACHE_DIR = Path(tempfile.gettempdir()) / "llm-inference-bench-plot-cache"
MATPLOTLIB_CACHE_DIR = PLOT_CACHE_DIR / "matplotlib"
XDG_CACHE_DIR = PLOT_CACHE_DIR / "xdg"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRECISIONS = ["fp16", "int8", "int4"]
BATCH_SIZES = [1, 8, 32]
PRECISION_INPUT_FILES = [
    OUTPUT_DIR / f"{precision}_batch{batch_size}.csv"
    for precision in PRECISIONS
    for batch_size in BATCH_SIZES
]
LEGACY_INPUT_FILES = [
    OUTPUT_DIR / "benchmark_results_b1.csv",
    OUTPUT_DIR / "benchmark_results_b8.csv",
]

REQUIRED_COLUMNS = {
    "batch_size",
    "latency_seconds",
    "throughput_tokens_per_sec",
}


def infer_precision_from_path(path: Path) -> str:
    for precision in PRECISIONS:
        if path.name.startswith(f"{precision}_"):
            return precision
    return "legacy"


def load_results(paths: list[Path]) -> pd.DataFrame:
    """Load benchmark CSVs and validate the expected metric columns."""
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        formatted = ", ".join(str(path) for path in paths)
        raise FileNotFoundError(f"No benchmark CSV files found. Expected: {formatted}")

    frames = []
    for path in existing_paths:
        frame = pd.read_csv(path)
        if "precision" not in frame.columns:
            frame = frame.assign(precision=infer_precision_from_path(path))
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)

    missing_columns = REQUIRED_COLUMNS.difference(results.columns)
    if missing_columns:
        formatted = ", ".join(sorted(missing_columns))
        raise ValueError(f"Benchmark CSVs are missing required column(s): {formatted}")

    return results


def select_inputs() -> tuple[list[Path], str]:
    """Prefer precision matrix outputs, with legacy b1/b8 fallback."""
    if any(path.exists() for path in PRECISION_INPUT_FILES):
        return PRECISION_INPUT_FILES, "precision"
    return LEGACY_INPUT_FILES, "legacy"


def aggregate_metrics(results: pd.DataFrame, by_precision: bool) -> pd.DataFrame:
    """Aggregate latency, throughput, and memory metrics."""
    if "gpu_memory_allocated_gb" not in results.columns:
        results = results.assign(gpu_memory_allocated_gb=0.0)
    if "gpu_memory_reserved_gb" not in results.columns:
        results = results.assign(gpu_memory_reserved_gb=0.0)

    group_columns = ["precision", "batch_size"] if by_precision else ["batch_size"]
    aggregated = (
        results.groupby(group_columns, as_index=False)
        .agg(
            mean_latency=("latency_seconds", "mean"),
            p95_latency=("latency_seconds", lambda values: values.quantile(0.95)),
            throughput=("throughput_tokens_per_sec", "mean"),
            max_gpu_memory_allocated_gb=("gpu_memory_allocated_gb", "max"),
            max_gpu_memory_reserved_gb=("gpu_memory_reserved_gb", "max"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return aggregated


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.5, 5.0),
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
        }
    )


def save_line_plot(
    aggregated: pd.DataFrame,
    y_column: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(
        aggregated["batch_size"],
        aggregated[y_column],
        marker="o",
        color="#1f77b4",
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(y_label)
    ax.set_xticks(aggregated["batch_size"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_precision_line_plot(
    aggregated: pd.DataFrame,
    y_column: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots()
    for precision in PRECISIONS:
        precision_rows = aggregated[aggregated["precision"] == precision]
        if precision_rows.empty:
            continue
        ax.plot(
            precision_rows["batch_size"],
            precision_rows[y_column],
            marker="o",
            label=precision,
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(y_label)
    ax.set_xticks(sorted(aggregated["batch_size"].unique()))
    ax.legend(title="Precision")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def print_summary_table(aggregated: pd.DataFrame) -> None:
    by_precision = "precision" in aggregated.columns
    if by_precision:
        header = (
            "Precision | Batch Size | Mean Latency | P95 Latency | "
            "Throughput | Max Memory"
        )
    else:
        header = "Batch Size | Mean Latency | P95 Latency | Throughput | Max Memory"
    print(f"\n{header}")
    print("-" * len(header))
    for row in aggregated.itertuples(index=False):
        if by_precision:
            print(
                f"{row.precision:>9} | "
                f"{int(row.batch_size):>10} | "
                f"{row.mean_latency:>12.4f} | "
                f"{row.p95_latency:>11.4f} | "
                f"{row.throughput:>10.4f} | "
                f"{row.max_gpu_memory_allocated_gb:>10.4f}"
            )
        else:
            print(
                f"{int(row.batch_size):>10} | "
                f"{row.mean_latency:>12.4f} | "
                f"{row.p95_latency:>11.4f} | "
                f"{row.throughput:>10.4f} | "
                f"{row.max_gpu_memory_allocated_gb:>10.4f}"
            )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    input_files, output_prefix = select_inputs()
    if output_prefix == "precision":
        aggregated_output = OUTPUT_DIR / "precision_aggregated_metrics.csv"
    else:
        aggregated_output = OUTPUT_DIR / "aggregated_metrics.csv"

    results = load_results(input_files)
    aggregated = aggregate_metrics(results, by_precision=output_prefix == "precision")
    aggregated.to_csv(aggregated_output, index=False)

    if output_prefix == "precision":
        save_precision_line_plot(
            aggregated=aggregated,
            y_column="throughput",
            y_label="Mean Throughput (tokens/sec)",
            title="Throughput vs Batch Size by Precision",
            output_path=OUTPUT_DIR / "precision_throughput_vs_batch.png",
        )
        save_precision_line_plot(
            aggregated=aggregated,
            y_column="mean_latency",
            y_label="Mean Latency (seconds)",
            title="Mean Latency vs Batch Size by Precision",
            output_path=OUTPUT_DIR / "precision_latency_vs_batch.png",
        )
        save_precision_line_plot(
            aggregated=aggregated,
            y_column="max_gpu_memory_allocated_gb",
            y_label="Max GPU Memory Allocated (GB)",
            title="GPU Memory vs Batch Size by Precision",
            output_path=OUTPUT_DIR / "precision_memory_vs_batch.png",
        )
    else:
        save_line_plot(
            aggregated=aggregated,
            y_column="throughput",
            y_label="Mean Throughput (tokens/sec)",
            title="Throughput vs Batch Size",
            output_path=OUTPUT_DIR / "throughput_vs_batch.png",
        )
        save_line_plot(
            aggregated=aggregated,
            y_column="mean_latency",
            y_label="Mean Latency (seconds)",
            title="Mean Latency vs Batch Size",
            output_path=OUTPUT_DIR / "latency_vs_batch.png",
        )
        save_line_plot(
            aggregated=aggregated,
            y_column="p95_latency",
            y_label="P95 Latency (seconds)",
            title="P95 Latency vs Batch Size",
            output_path=OUTPUT_DIR / "p95_latency_vs_batch.png",
        )

    print_summary_table(aggregated)
    print(f"\nSaved aggregated metrics to {aggregated_output}")
    print(f"Saved plots to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
