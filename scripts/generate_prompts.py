"""Generate a fixed prompt set for the FP16 benchmark experiments."""

from __future__ import annotations

import json
from pathlib import Path


SHORT_PROMPTS = [
    "Summarize why GPUs are useful for neural network inference.",
    "Write a two-sentence explanation of batch size.",
    "List three common metrics for an inference benchmark.",
    "Explain the difference between latency and throughput.",
    "Give one practical tip for reducing LLM serving cost.",
    "Define FP16 inference in one paragraph.",
    "Name two reasons to run warmup iterations before timing.",
    "Explain why output length affects benchmark results.",
    "State one risk of comparing benchmarks across different GPUs.",
    "Describe what tokens per second measures.",
]

MEDIUM_PROMPTS = [
    (
        "You are reviewing an internal benchmark report. Explain how prompt "
        "length, generated token count, and batch size can each affect measured "
        "tokens per second."
    ),
    (
        "Draft a concise checklist for validating a new language model inference "
        "pipeline before publishing results to a team dashboard."
    ),
    (
        "Compare CPU fallback and CUDA execution for a small transformer model. "
        "Focus on reliability, expected speed, and memory behavior."
    ),
    (
        "Explain why warmup runs are useful in GPU benchmarks and what could go "
        "wrong if they are skipped."
    ),
    (
        "Write a short project update describing progress on an FP16 baseline "
        "for an LLM inference benchmark."
    ),
    (
        "Describe how batching can improve aggregate throughput while still "
        "creating latency tradeoffs in an online inference service."
    ),
    (
        "Explain how deterministic generation settings help make benchmark "
        "runs easier to compare across repeated experiments."
    ),
    (
        "Write a short note for a teammate explaining why benchmark CSV files "
        "should include model settings, batch size, and output token counts."
    ),
    (
        "Compare mean latency and p95 latency as ways to summarize LLM inference "
        "performance for a small experiment."
    ),
    (
        "Explain how GPU memory allocation and reserved memory can help diagnose "
        "whether a benchmark is close to running out of memory."
    ),
]

LONG_PROMPTS = [
    (
        "Create a structured analysis of an LLM inference benchmark that uses "
        "HuggingFace Transformers. Discuss model loading, prompt preparation, "
        "batching, timing boundaries, GPU synchronization, memory measurement, "
        "CSV output, and how the baseline could be extended later."
    ),
    (
        "You are designing a portfolio project for benchmarking local LLM "
        "inference. Describe the minimum viable implementation, the risks of "
        "overengineering early, and a practical roadmap that adds quantization "
        "and serving engines after the baseline is trustworthy."
    ),
    (
        "Write a technical note explaining how generated tokens per second should "
        "be interpreted. Include caveats about different prompt lengths, batch "
        "sizes, hardware, model precision, tokenizer behavior, and deterministic "
        "generation settings."
    ),
    (
        "Prepare an onboarding explanation for a teammate who will run an FP16 "
        "LLM benchmark for the first time. Cover environment setup, HuggingFace "
        "authentication, CUDA availability, expected outputs, and common failure "
        "modes."
    ),
    (
        "Analyze the tradeoffs between measuring per-prompt latency and per-batch "
        "latency in a simple benchmark. Explain how each view can be useful, what "
        "the CSV rows should represent, and how future time-to-first-token metrics "
        "could improve the report."
    ),
    (
        "Write a design memo for extending an FP16 benchmark into a broader "
        "serving evaluation. Include batch scaling, quantization, backend "
        "comparison, reproducibility concerns, and what should be measured before "
        "claiming production readiness."
    ),
    (
        "Explain why a benchmark should report both raw per-run data and "
        "aggregated plots. Discuss debugging value, statistical interpretation, "
        "communication with stakeholders, and how raw data protects against "
        "misleading summaries."
    ),
    (
        "Create a practical guide for running an LLM inference benchmark in "
        "Google Colab. Mention runtime selection, dependency installation, model "
        "access, GPU memory constraints, output artifacts, and how to preserve "
        "results after the session ends."
    ),
    (
        "Compare three approaches to improving LLM inference efficiency: batching, "
        "quantization, and optimized serving engines. Explain what each approach "
        "changes, what metrics it can improve, and what risks should be measured."
    ),
    (
        "Draft a results section for a systems performance writeup where batch "
        "size is increased from one to larger values. Explain how to discuss "
        "throughput gains, latency movement, p95 latency, GPU memory use, and "
        "limitations without overstating the findings."
    ),
]


def _build_prompt_items() -> list[dict[str, str]]:
    items = []
    for category, prompts in (
        ("short", SHORT_PROMPTS),
        ("medium", MEDIUM_PROMPTS),
        ("long", LONG_PROMPTS),
    ):
        for index, prompt in enumerate(prompts, start=1):
            items.append(
                {
                    "id": f"{category}-{index:03d}",
                    "category": category,
                    "prompt": prompt,
                }
            )
    return items


PROMPTS = _build_prompt_items()


def main() -> None:
    output_path = Path("data/prompts.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(PROMPTS, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(PROMPTS)} prompts to {output_path}")


if __name__ == "__main__":
    main()
