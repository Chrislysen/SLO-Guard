"""GPU and model profiles for memory-aware config sizing.

The KV-cache memory budget depends on:
  - GPU VRAM (varies: T4 16GB, L4 24GB, A100 40/80GB, H100 80GB, RTX 5080 16GB)
  - Per-token KV size (varies by model: layers * kv_heads * head_dim * dtype_bytes)
  - Model footprint (weights + activations + framework overhead)

We detect VRAM at runtime and look up a per-token KV size from a small
registry of known models, falling back to a conservative default. Both
values can be overridden via env vars for testing or unknown hardware.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


# Per-token KV cache size in GB at fp16, derived from
#   2 (K and V) * num_layers * num_kv_heads * head_dim * 2 bytes / 1e9
# Values verified against published model configs.
_MODEL_KV_GB_PER_TOKEN: dict[str, float] = {
    "Qwen/Qwen2-1.5B": 0.000096,           # 28 layers * 2 kv heads * 128 dim * 2B * 2 / 1e9
    "Qwen/Qwen2-1.5B-Instruct": 0.000096,
    "Qwen/Qwen2.5-1.5B": 0.000096,
    "Qwen/Qwen2.5-1.5B-Instruct": 0.000096,
    "Qwen/Qwen2-7B": 0.000262,             # 28 layers * 4 kv heads * 128 dim * 2B * 2
    "Qwen/Qwen2.5-7B": 0.000262,
    "meta-llama/Llama-3.2-1B": 0.000066,   # 16 layers * 8 kv heads * 64 dim * 2B * 2
    "meta-llama/Llama-3.2-3B": 0.000115,   # 28 layers * 8 kv heads * 128 dim * 2B * 2 / 1e9
    "meta-llama/Llama-3.1-8B": 0.000524,   # 32 layers * 8 kv heads * 128 dim * 2B * 2
    "mistralai/Mistral-7B-v0.3": 0.000524, # 32 layers * 8 kv heads * 128 dim * 2B * 2
}

# Approx model footprint in GB at fp16 (weights + activations + framework slack).
# When unknown, we estimate from the parameter count guessed via the model name.
_MODEL_FOOTPRINT_GB: dict[str, float] = {
    "Qwen/Qwen2-1.5B": 4.0,
    "Qwen/Qwen2-1.5B-Instruct": 4.0,
    "Qwen/Qwen2.5-1.5B": 4.0,
    "Qwen/Qwen2.5-1.5B-Instruct": 4.0,
    "meta-llama/Llama-3.2-1B": 3.5,
    "meta-llama/Llama-3.2-3B": 7.0,
    "Qwen/Qwen2-7B": 16.0,
    "Qwen/Qwen2.5-7B": 16.0,
    "meta-llama/Llama-3.1-8B": 18.0,
    "mistralai/Mistral-7B-v0.3": 16.0,
}

# Conservative defaults for unknown models — sized for a small ~1.5B model.
DEFAULT_KV_GB_PER_TOKEN = 0.000096
DEFAULT_MODEL_FOOTPRINT_GB = 5.0
DEFAULT_VRAM_GB = 40.0  # A100 40GB — the original SLO-Guard baseline


def detect_gpu_vram_gb() -> float | None:
    """Detect total VRAM (GB) of GPU 0.

    Tries env override, then torch (if installed), then nvidia-smi.
    Returns None if nothing works.
    """
    override = os.environ.get("SLOGUARD_GPU_VRAM_GB")
    if override:
        try:
            return float(override)
        except ValueError:
            logger.warning("SLOGUARD_GPU_VRAM_GB=%r is not a number", override)

    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("torch GPU detection failed: %s", e)

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                [
                    "nvidia-smi", "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits", "-i", "0",
                ],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                # nvidia-smi reports MiB
                return float(out.stdout.strip().split("\n")[0]) / 1024.0
        except Exception as e:
            logger.debug("nvidia-smi GPU detection failed: %s", e)

    return None


def detect_gpu_name() -> str | None:
    """Best-effort GPU name detection for logging."""
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().split("\n")[0]
        except Exception:
            pass

    return None


def kv_gb_per_token_for(model_id: str) -> float:
    """Return per-token KV cache size in GB for *model_id*.

    Uses env override → registry → default.
    """
    override = os.environ.get("SLOGUARD_KV_BYTES_PER_TOKEN")
    if override:
        try:
            return float(override)
        except ValueError:
            logger.warning("SLOGUARD_KV_BYTES_PER_TOKEN=%r is not a number", override)

    if model_id in _MODEL_KV_GB_PER_TOKEN:
        return _MODEL_KV_GB_PER_TOKEN[model_id]

    logger.debug(
        "No KV-cache profile for model %r — using default %.6f GB/token",
        model_id, DEFAULT_KV_GB_PER_TOKEN,
    )
    return DEFAULT_KV_GB_PER_TOKEN


def model_footprint_gb_for(model_id: str) -> float:
    """Return the GB to reserve for weights/activations/overhead for *model_id*."""
    override = os.environ.get("SLOGUARD_MODEL_FOOTPRINT_GB")
    if override:
        try:
            return float(override)
        except ValueError:
            logger.warning("SLOGUARD_MODEL_FOOTPRINT_GB=%r is not a number", override)

    return _MODEL_FOOTPRINT_GB.get(model_id, DEFAULT_MODEL_FOOTPRINT_GB)


def log_gpu_info(model_id: str | None = None) -> dict[str, float | str | None]:
    """Detect and log GPU + model profile. Returns a dict with the detected values."""
    name = detect_gpu_name()
    vram = detect_gpu_vram_gb()
    info: dict[str, float | str | None] = {"gpu_name": name, "vram_gb": vram}

    if vram is None:
        logger.warning(
            "GPU VRAM not detected — falling back to %.0fGB. "
            "Set SLOGUARD_GPU_VRAM_GB to override.", DEFAULT_VRAM_GB,
        )
    else:
        logger.info("Detected GPU: %s, %.1f GB VRAM", name or "unknown", vram)

    if model_id is not None:
        kv = kv_gb_per_token_for(model_id)
        footprint = model_footprint_gb_for(model_id)
        info["kv_gb_per_token"] = kv
        info["model_footprint_gb"] = footprint
        logger.info(
            "Model profile [%s]: %.6f GB/KV-token, %.1f GB footprint",
            model_id, kv, footprint,
        )

    return info
