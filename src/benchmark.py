"""Benchmark execution logic for Transformers generation."""

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


def run_single_generation(model, tokenizer, prompt: str, max_new_tokens: int):
    """Generate from one prompt and return output metadata."""

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_tokens = int(output_ids.shape[-1])
    generated_tokens = max(output_tokens - prompt_tokens, 0)
    return {
        "prompt": prompt,
        "prompt_length_tokens": prompt_tokens,
        "output_length_tokens": output_tokens,
        "generated_tokens": generated_tokens,
        "text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
    }


def _run_generation_batch(model, tokenizer, prompts: list[str], max_new_tokens: int):
    device = next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    padded_input_length = int(inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    rows = []
    for index, prompt in enumerate(prompts):
        prompt_length = int(prompt_lengths[index])
        generated_tokens = max(int(output_ids[index].shape[-1]) - padded_input_length, 0)
        output_length = prompt_length + generated_tokens
        rows.append(
            {
                "prompt": prompt,
                "prompt_length_tokens": prompt_length,
                "output_length_tokens": output_length,
                "generated_tokens": generated_tokens,
            }
        )
    return rows


def run_benchmark(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    warmup_runs: int,
    num_runs: int,
):
    """Run batched generation and collect one result row per prompt per run."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not prompts:
        raise ValueError("prompts must contain at least one prompt")

    warmup_batches = list(_batched(prompts, batch_size))
    for warmup_index in tqdm(range(warmup_runs), desc="Warmup", leave=False):
        batch = warmup_batches[warmup_index % len(warmup_batches)]
        _run_generation_batch(model, tokenizer, batch, max_new_tokens)

    rows = []
    batches = list(_batched(prompts, batch_size))
    for run_index in tqdm(range(num_runs), desc="Benchmark runs"):
        for batch_index, batch in enumerate(tqdm(batches, desc="Batches", leave=False)):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            _sync_cuda()
            start = time.perf_counter()
            batch_rows = _run_generation_batch(model, tokenizer, batch, max_new_tokens)
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
                    }
                )

    return rows
