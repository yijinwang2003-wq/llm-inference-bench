"""Benchmark execution logic for the vLLM FP16 backend."""

from __future__ import annotations

import time
from itertools import islice
from typing import Iterable

import torch
from tqdm import tqdm


def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def _cuda_memory_gb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.max_memory_allocated() / (1024**3)
    reserved = torch.cuda.max_memory_reserved() / (1024**3)
    return allocated, reserved


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_vllm_model(model_name: str):
    """Load a vLLM FP16 model.

    vLLM is CUDA-oriented for this project phase. The import stays inside this
    function so non-vLLM workflows can still import the package modules.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("vLLM benchmark requires a CUDA GPU runtime.")

    from vllm import LLM

    return LLM(model=model_name, dtype="float16")


def _prompt_token_count(llm, prompt: str) -> int:
    tokenizer = llm.get_tokenizer()
    return len(tokenizer.encode(prompt))


def _run_generation_batch(llm, sampling_params, prompts: list[str]) -> list[dict]:
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    rows = []
    for prompt, output in zip(prompts, outputs, strict=True):
        completion = output.outputs[0] if output.outputs else None
        generated_tokens = len(completion.token_ids) if completion else 0
        prompt_tokens = (
            len(output.prompt_token_ids)
            if output.prompt_token_ids is not None
            else _prompt_token_count(llm, prompt)
        )
        rows.append(
            {
                "prompt": prompt,
                "prompt_length_tokens": prompt_tokens,
                "output_length_tokens": prompt_tokens + generated_tokens,
                "generated_tokens": generated_tokens,
            }
        )
    return rows


def run_vllm_benchmark(
    llm,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    warmup_runs: int,
    num_runs: int,
) -> list[dict]:
    """Run vLLM generation and collect one result row per prompt per run."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not prompts:
        raise ValueError("prompts must contain at least one prompt")

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    warmup_batches = list(_batched(prompts, batch_size))
    for warmup_index in tqdm(range(warmup_runs), desc="Warmup", leave=False):
        batch = warmup_batches[warmup_index % len(warmup_batches)]
        _run_generation_batch(llm, sampling_params, batch)

    rows = []
    batches = list(_batched(prompts, batch_size))
    for run_index in tqdm(range(num_runs), desc="Benchmark runs"):
        for batch_index, batch in enumerate(tqdm(batches, desc="Batches", leave=False)):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            _sync_cuda()
            start = time.perf_counter()
            batch_rows = _run_generation_batch(llm, sampling_params, batch)
            _sync_cuda()
            latency_seconds = time.perf_counter() - start

            total_generated_tokens = sum(row["generated_tokens"] for row in batch_rows)
            throughput = (
                total_generated_tokens / latency_seconds if latency_seconds > 0 else 0.0
            )
            allocated_gb, reserved_gb = _cuda_memory_gb()

            for prompt_index, row in enumerate(batch_rows):
                rows.append(
                    {
                        "run_index": run_index,
                        "batch_index": batch_index,
                        "batch_size": batch_size,
                        "prompt_index": batch_index * batch_size + prompt_index,
                        "prompt": row["prompt"],
                        "latency_seconds": latency_seconds,
                        "generated_tokens": row["generated_tokens"],
                        "throughput_tokens_per_sec": throughput,
                        "gpu_memory_allocated_gb": allocated_gb,
                        "gpu_memory_reserved_gb": reserved_gb,
                        "prompt_length_tokens": row["prompt_length_tokens"],
                        "output_length_tokens": row["output_length_tokens"],
                        "backend": "vllm_fp16",
                        "precision": "fp16",
                    }
                )

    return rows
