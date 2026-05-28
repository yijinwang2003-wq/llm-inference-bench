"""Metrics helpers for benchmark result rows."""

from __future__ import annotations

from statistics import mean


def percentile(values: list[float], p: float) -> float:
    """Return the pth percentile using linear interpolation."""

    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    rank = (p / 100) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def summarize_results(rows: list[dict]) -> dict[str, float]:
    """Compute summary metrics from benchmark rows."""

    if not rows:
        return {
            "mean_latency_seconds": 0.0,
            "p50_latency_seconds": 0.0,
            "p95_latency_seconds": 0.0,
            "mean_throughput_tokens_per_sec": 0.0,
        }

    latencies = [float(row["latency_seconds"]) for row in rows]
    throughputs = [float(row["throughput_tokens_per_sec"]) for row in rows]
    return {
        "mean_latency_seconds": mean(latencies),
        "p50_latency_seconds": percentile(latencies, 50),
        "p95_latency_seconds": percentile(latencies, 95),
        "mean_throughput_tokens_per_sec": mean(throughputs),
    }
