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


FP16_INPUT_FILES = [
    OUTPUT_DIR / "fp16_batch1.csv",
    OUTPUT_DIR / "fp16_batch8.csv",
    OUTPUT_DIR / "fp16_batch32.csv",
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


def load_results(paths: list[Path]) -> pd.DataFrame:
    """Load benchmark CSVs and validate the expected metric columns."""
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        formatted = ", ".join(str(path) for path in paths)
        raise FileNotFoundError(f"No benchmark CSV files found. Expected: {formatted}")

    frames = [pd.read_csv(path) for path in existing_paths]
    results = pd.concat(frames, ignore_index=True)

    missing_columns = REQUIRED_COLUMNS.difference(results.columns)
    if missing_columns:
        formatted = ", ".join(sorted(missing_columns))
        raise ValueError(f"Benchmark CSVs are missing required column(s): {formatted}")

    return results


def select_inputs() -> tuple[list[Path], str]:
    """Prefer the full FP16 matrix outputs, with legacy b1/b8 fallback."""
    if any(path.exists() for path in FP16_INPUT_FILES):
        return FP16_INPUT_FILES, "fp16"
    return LEGACY_INPUT_FILES, "legacy"


def aggregate_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate latency and throughput metrics by batch size."""
    if "gpu_memory_allocated_gb" not in results.columns:
        results = results.assign(gpu_memory_allocated_gb=0.0)
    if "gpu_memory_reserved_gb" not in results.columns:
        results = results.assign(gpu_memory_reserved_gb=0.0)

    aggregated = (
        results.groupby("batch_size", as_index=False)
        .agg(
            mean_latency=("latency_seconds", "mean"),
            p95_latency=("latency_seconds", lambda values: values.quantile(0.95)),
            throughput=("throughput_tokens_per_sec", "mean"),
            max_gpu_memory_allocated_gb=("gpu_memory_allocated_gb", "max"),
            max_gpu_memory_reserved_gb=("gpu_memory_reserved_gb", "max"),
        )
        .sort_values("batch_size")
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


def print_summary_table(aggregated: pd.DataFrame) -> None:
    header = "Batch Size | Mean Latency | P95 Latency | Throughput | Max Memory"
    print(f"\n{header}")
    print("-" * len(header))
    for row in aggregated.itertuples(index=False):
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
    if output_prefix == "fp16":
        aggregated_output = OUTPUT_DIR / "fp16_aggregated_metrics.csv"
        throughput_plot = OUTPUT_DIR / "fp16_throughput_vs_batch.png"
        mean_latency_plot = OUTPUT_DIR / "fp16_mean_latency_vs_batch.png"
        p95_latency_plot = OUTPUT_DIR / "fp16_p95_latency_vs_batch.png"
        memory_plot = OUTPUT_DIR / "fp16_memory_vs_batch.png"
    else:
        aggregated_output = OUTPUT_DIR / "aggregated_metrics.csv"
        throughput_plot = OUTPUT_DIR / "throughput_vs_batch.png"
        mean_latency_plot = OUTPUT_DIR / "latency_vs_batch.png"
        p95_latency_plot = OUTPUT_DIR / "p95_latency_vs_batch.png"
        memory_plot = None

    results = load_results(input_files)
    aggregated = aggregate_metrics(results)
    aggregated.to_csv(aggregated_output, index=False)

    save_line_plot(
        aggregated=aggregated,
        y_column="throughput",
        y_label="Mean Throughput (tokens/sec)",
        title="Throughput vs Batch Size",
        output_path=throughput_plot,
    )
    save_line_plot(
        aggregated=aggregated,
        y_column="mean_latency",
        y_label="Mean Latency (seconds)",
        title="Mean Latency vs Batch Size",
        output_path=mean_latency_plot,
    )
    save_line_plot(
        aggregated=aggregated,
        y_column="p95_latency",
        y_label="P95 Latency (seconds)",
        title="P95 Latency vs Batch Size",
        output_path=p95_latency_plot,
    )
    if memory_plot is not None:
        save_line_plot(
            aggregated=aggregated,
            y_column="max_gpu_memory_allocated_gb",
            y_label="Max GPU Memory Allocated (GB)",
            title="GPU Memory vs Batch Size",
            output_path=memory_plot,
        )

    print_summary_table(aggregated)
    print(f"\nSaved aggregated metrics to {aggregated_output}")
    print(f"Saved plots to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
