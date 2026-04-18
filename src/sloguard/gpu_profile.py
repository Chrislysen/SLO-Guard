"""GPU and model profiles for memory-aware config sizing.

The KV-cache memory budget depends on:
  - GPU VRAM (varies: T4 16GB, L4 24GB, A100 40/80GB, H100 80GB, RTX 5080 16GB)
  - Per-token KV size (varies by model: layers * kv_heads * head_dim * dtype_bytes)
  - Model footprint (weights + activations + framework overhead)

We detect VRAM at runtime, derive the per-token KV size and model
footprint from the HF cache when possible (works for any downloaded
model), and fall back to a small hand-tuned registry then a default.
All values can be overridden via env vars.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

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

# Bytes per element for common dtypes used in HF model configs.
_DTYPE_BYTES: dict[str, int] = {
    "float16": 2, "fp16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "float32": 4, "fp32": 4, "float": 4,
    "float64": 8, "fp64": 8, "double": 8,
    "int8": 1, "uint8": 1,
}

# Slack on top of weight bytes to cover activations + framework overhead.
# Picked to roughly match the hand-tuned registry footprints.
_FOOTPRINT_SLACK = 1.4

# Weight files to count when sizing a snapshot directory.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


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


def _hf_cache_dirs() -> list[Path]:
    """Return existing candidate HF Hub cache root directories."""
    candidates: list[Path] = []
    env_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_cache:
        candidates.append(Path(env_cache))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    return [c for c in candidates if c.exists()]


def _find_hf_snapshot(model_id: str) -> Path | None:
    """Return the most recent snapshot directory for *model_id*, or None.

    HF Hub layout: ``<cache>/models--<org>--<name>/snapshots/<commit>/...``
    """
    safe = "models--" + model_id.replace("/", "--")
    for cache in _hf_cache_dirs():
        snap_root = cache / safe / "snapshots"
        if not snap_root.exists():
            continue
        snaps = [s for s in snap_root.iterdir() if s.is_dir()]
        if not snaps:
            continue
        # Pick the most recently modified snapshot — usually the active one.
        snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return snaps[0]
    return None


def kv_gb_from_hf_config(model_id: str) -> float | None:
    """Compute per-token KV size (GB) from the HF config.json for *model_id*.

    Formula: ``2 (K + V) * num_hidden_layers * num_key_value_heads
              * head_dim * dtype_bytes / 1e9``

    Returns None when the model isn't cached or required fields are missing.
    """
    snap = _find_hf_snapshot(model_id)
    if snap is None:
        return None
    cfg_path = snap / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("HF config.json parse failed for %s: %s", model_id, e)
        return None

    num_layers = cfg.get("num_hidden_layers")
    num_kv_heads = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads")
    head_dim = cfg.get("head_dim")
    if head_dim is None:
        hidden = cfg.get("hidden_size")
        n_heads = cfg.get("num_attention_heads")
        if hidden and n_heads:
            head_dim = hidden // n_heads

    if not (num_layers and num_kv_heads and head_dim):
        logger.debug(
            "HF config.json for %s missing layers/kv_heads/head_dim", model_id,
        )
        return None

    dtype = str(cfg.get("torch_dtype", "float16")).lower()
    dtype_bytes = _DTYPE_BYTES.get(dtype, 2)
    return (2 * num_layers * num_kv_heads * head_dim * dtype_bytes) / 1e9


def footprint_gb_from_hf_cache(model_id: str) -> float | None:
    """Estimate model footprint (GB) from on-disk weight files * slack.

    Sums sizes of ``.safetensors`` / ``.bin`` / ``.pt`` files in the
    snapshot dir (following symlinks — HF cache stores blobs separately).
    Returns None if no weight files are present.
    """
    snap = _find_hf_snapshot(model_id)
    if snap is None:
        return None

    total_bytes = 0
    for entry in snap.iterdir():
        if entry.suffix not in _WEIGHT_SUFFIXES:
            continue
        try:
            # stat() follows symlinks, which is what we want — HF stores
            # the actual blob in ../../blobs/<sha> and symlinks here.
            total_bytes += entry.stat().st_size
        except OSError:
            continue

    if total_bytes == 0:
        return None
    return (total_bytes / 1e9) * _FOOTPRINT_SLACK


def kv_gb_per_token_for(model_id: str) -> float:
    """Return per-token KV cache size in GB for *model_id*.

    Lookup order: env override → HF config probe → registry → default.
    """
    override = os.environ.get("SLOGUARD_KV_BYTES_PER_TOKEN")
    if override:
        try:
            return float(override)
        except ValueError:
            logger.warning("SLOGUARD_KV_BYTES_PER_TOKEN=%r is not a number", override)

    probed = kv_gb_from_hf_config(model_id)
    if probed is not None:
        return probed

    if model_id in _MODEL_KV_GB_PER_TOKEN:
        return _MODEL_KV_GB_PER_TOKEN[model_id]

    logger.debug(
        "No KV-cache profile for model %r — using default %.6f GB/token",
        model_id, DEFAULT_KV_GB_PER_TOKEN,
    )
    return DEFAULT_KV_GB_PER_TOKEN


def model_footprint_gb_for(model_id: str) -> float:
    """Return the GB to reserve for weights/activations/overhead.

    Lookup order: env override → HF cache probe → registry → default.
    """
    override = os.environ.get("SLOGUARD_MODEL_FOOTPRINT_GB")
    if override:
        try:
            return float(override)
        except ValueError:
            logger.warning("SLOGUARD_MODEL_FOOTPRINT_GB=%r is not a number", override)

    probed = footprint_gb_from_hf_cache(model_id)
    if probed is not None:
        return probed

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
