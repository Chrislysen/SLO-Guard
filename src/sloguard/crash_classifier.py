"""Crash classifier for vLLM server failures.

Pattern-matches server stderr and exit codes to classify crash types.
Crashes are data — structured classification enables the feasibility
model to learn which config regions are dangerous.
"""
from __future__ import annotations

import re
from enum import Enum


class CrashType(str, Enum):
    """Classification of serving failure modes."""
    HEALTHY = "healthy"
    OOM = "oom"
    CUDA_ERROR = "cuda_error"
    TIMEOUT = "timeout"
    CONFIG_INVALID = "config_invalid"
    STARTUP_FAILURE = "startup_failure"
    UNKNOWN = "unknown"


# Patterns ordered by specificity (most specific first)
_OOM_PATTERNS = [
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"OutOfMemoryError", re.IGNORECASE),
    re.compile(r"torch\.cuda\.OutOfMemoryError", re.IGNORECASE),
    re.compile(r"CUDA error: out of memory", re.IGNORECASE),
    re.compile(r"Memory allocation failed", re.IGNORECASE),
    re.compile(r"OOM", re.IGNORECASE),
    re.compile(r"Cannot allocate memory", re.IGNORECASE),
    re.compile(r"not enough memory", re.IGNORECASE),
    re.compile(r"KV cache .* too small", re.IGNORECASE),
    re.compile(r"gpu_memory_utilization .* is too high", re.IGNORECASE),
    re.compile(r"insufficient memory", re.IGNORECASE),
]

_CUDA_ERROR_PATTERNS = [
    re.compile(r"CUDA error:", re.IGNORECASE),
    re.compile(r"cudaError", re.IGNORECASE),
    re.compile(r"CUBLAS_STATUS", re.IGNORECASE),
    re.compile(r"CUSOLVER_STATUS", re.IGNORECASE),
    re.compile(r"NCCL error", re.IGNORECASE),
    re.compile(r"RuntimeError:.*CUDA", re.IGNORECASE),
    re.compile(r"cuBLAS error", re.IGNORECASE),
    re.compile(r"CUDA driver error", re.IGNORECASE),
    re.compile(r"illegal memory access", re.IGNORECASE),
]

_CONFIG_INVALID_PATTERNS = [
    re.compile(r"ValueError:", re.IGNORECASE),
    re.compile(r"quantization .* not supported", re.IGNORECASE),
    re.compile(r"Unsupported .* configuration", re.IGNORECASE),
    re.compile(r"Invalid .* argument", re.IGNORECASE),
    re.compile(r"max_num_batched_tokens .* must be", re.IGNORECASE),
    re.compile(r"block_size must be", re.IGNORECASE),
    re.compile(r"Cannot use .* with", re.IGNORECASE),
    re.compile(r"Incompatible .* options", re.IGNORECASE),
    re.compile(r"model .* does not support", re.IGNORECASE),
    re.compile(r"is not compatible with", re.IGNORECASE),
]

_STARTUP_FAILURE_PATTERNS = [
    re.compile(r"Failed to start", re.IGNORECASE),
    re.compile(r"Server failed", re.IGNORECASE),
    re.compile(r"ModuleNotFoundError", re.IGNORECASE),
    re.compile(r"ImportError", re.IGNORECASE),
    re.compile(r"FileNotFoundError", re.IGNORECASE),
    re.compile(r"Connection refused", re.IGNORECASE),
    re.compile(r"Address already in use", re.IGNORECASE),
    re.compile(r"vLLM not found", re.IGNORECASE),
]


class CrashClassifier:
    """Classifies vLLM server failures from stderr output and exit info.

    Usage:
        classifier = CrashClassifier()
        crash_type = classifier.classify(
            stderr="CUDA out of memory. Tried to allocate...",
            exit_code=-9,
            timed_out=False,
        )
        # Returns CrashType.OOM
    """

    def classify(
        self,
        stderr: str = "",
        exit_code: int | None = None,
        timed_out: bool = False,
        exception: Exception | None = None,
    ) -> CrashType:
        """Classify a server failure.

        Args:
            stderr: Captured stderr output from the vLLM process.
            exit_code: Process exit code (None if still running / healthy).
            timed_out: Whether the server startup or benchmark timed out.
            exception: Any Python exception caught during evaluation.

        Returns:
            CrashType enum value.
        """
        # Healthy: no error indicators
        if not stderr and exit_code is None and not timed_out and exception is None:
            return CrashType.HEALTHY

        if exit_code == 0 and not timed_out and exception is None:
            return CrashType.HEALTHY

        # Timeout takes priority (may mask other errors)
        if timed_out:
            return CrashType.TIMEOUT

        # Combine all text sources for pattern matching
        text = stderr
        if exception is not None:
            text += f"\n{type(exception).__name__}: {exception}"

        # Check OOM first (most common in serving)
        for pattern in _OOM_PATTERNS:
            if pattern.search(text):
                return CrashType.OOM

        # CUDA errors
        for pattern in _CUDA_ERROR_PATTERNS:
            if pattern.search(text):
                return CrashType.CUDA_ERROR

        # Config validation errors
        for pattern in _CONFIG_INVALID_PATTERNS:
            if pattern.search(text):
                return CrashType.CONFIG_INVALID

        # Startup failures
        for pattern in _STARTUP_FAILURE_PATTERNS:
            if pattern.search(text):
                return CrashType.STARTUP_FAILURE

        # OOM by exit code (killed by OS OOM killer)
        if exit_code is not None and exit_code in (-9, 137):
            return CrashType.OOM

        # Segfault
        if exit_code is not None and exit_code in (-11, 139):
            return CrashType.CUDA_ERROR

        # Any non-zero exit
        if exit_code is not None and exit_code != 0:
            return CrashType.UNKNOWN

        return CrashType.UNKNOWN

    def classify_exception(self, exc: Exception) -> CrashType:
        """Convenience: classify from a caught exception."""
        return self.classify(exception=exc)
