"""Model loading utilities for Transformers inference baselines."""

from __future__ import annotations

from importlib.metadata import version


SUPPORTED_PRECISIONS = ("fp16", "int8", "int4")


def _torch_version_tuple() -> tuple[int, ...]:
    raw_version = version("torch").split("+", maxsplit=1)[0]
    parts = []
    for part in raw_version.split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts)


def _is_auth_or_access_error(error: Exception) -> bool:
    message = str(error).lower()
    access_markers = (
        "401",
        "403",
        "gated",
        "restricted",
        "unauthorized",
        "forbidden",
        "access token",
        "must be authenticated",
        "not authorized",
    )
    return any(marker in message for marker in access_markers)


def _is_network_error(error: Exception) -> bool:
    message = str(error).lower()
    network_markers = (
        "nodename nor servname provided",
        "name or service not known",
        "temporary failure in name resolution",
        "connection error",
        "connection refused",
        "network is unreachable",
        "read timed out",
    )
    return any(marker in message for marker in network_markers)


def load_model(model_name: str, precision: str = "fp16"):
    """Load a causal language model for the requested precision.

    FP16 uses CUDA float16 when available and CPU float32 fallback for smoke
    tests. INT8 and INT4 use bitsandbytes and require CUDA.
    """

    if precision not in SUPPORTED_PRECISIONS:
        formatted = ", ".join(SUPPORTED_PRECISIONS)
        raise ValueError(f"precision must be one of: {formatted}")

    if _torch_version_tuple() < (2, 4):
        raise RuntimeError(
            f"Installed torch version is {version('torch')}, but this benchmark "
            "requires torch >= 2.4 for the installed Transformers package. Run:\n\n"
            "pip install -r requirements.txt"
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    if precision in {"int8", "int4"} and not cuda_available:
        raise RuntimeError(
            f"{precision} quantization requires CUDA and bitsandbytes. "
            "Run quantized benchmarks on a Colab GPU or another CUDA runtime."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if precision == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        elif precision == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        elif cuda_available:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
    except Exception as exc:
        if _is_auth_or_access_error(exc):
            raise RuntimeError(
                "Unable to access the HuggingFace model. Llama models may require "
                "license acceptance and authentication. Run:\n\n"
                "huggingface-cli login\n\n"
                f"Then retry loading {model_name}."
            ) from exc
        if _is_network_error(exc):
            raise RuntimeError(
                "Unable to reach HuggingFace to download the model. Check network "
                "connectivity and retry. If the model is gated, also run:\n\n"
                "huggingface-cli login"
            ) from exc
        raise RuntimeError(f"Failed to load model '{model_name}': {exc}") from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if not cuda_available:
        model.to(device)
    model.eval()
    return model, tokenizer, device


def load_model_fp16(model_name: str):
    """Load a causal language model for the FP16 baseline."""

    return load_model(model_name, precision="fp16")
