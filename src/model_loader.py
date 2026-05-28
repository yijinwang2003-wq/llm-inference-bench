"""Model loading utilities for the FP16 Transformers baseline."""

from __future__ import annotations

from importlib.metadata import version


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


def load_model_fp16(model_name: str):
    """Load a causal language model for the FP16 baseline.

    CUDA uses float16. CPU fallback uses float32 because CPU float16 generation is
    poorly supported and often slower or numerically fragile.
    """

    if _torch_version_tuple() < (2, 4):
        raise RuntimeError(
            f"Installed torch version is {version('torch')}, but this benchmark "
            "requires torch >= 2.4 for the installed Transformers package. Run:\n\n"
            "pip install -r requirements.txt"
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
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

    model.to(device)
    model.eval()
    return model, tokenizer, device
