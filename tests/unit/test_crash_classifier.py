"""Tests for crash classification against known error strings."""
from __future__ import annotations

import pytest

from sloguard.crash_classifier import CrashClassifier, CrashType


@pytest.fixture
def classifier():
    return CrashClassifier()


class TestCrashClassifier:
    def test_healthy_no_errors(self, classifier):
        result = classifier.classify(stderr="", exit_code=0)
        assert result == CrashType.HEALTHY

    def test_healthy_none_inputs(self, classifier):
        result = classifier.classify()
        assert result == CrashType.HEALTHY

    def test_oom_cuda(self, classifier):
        stderr = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
        result = classifier.classify(stderr=stderr, exit_code=1)
        assert result == CrashType.OOM

    def test_oom_torch(self, classifier):
        stderr = "torch.cuda.OutOfMemoryError: CUDA out of memory"
        result = classifier.classify(stderr=stderr, exit_code=1)
        assert result == CrashType.OOM

    def test_oom_kv_cache(self, classifier):
        stderr = "ValueError: KV cache is too small to hold all requests"
        result = classifier.classify(stderr=stderr, exit_code=1)
        assert result == CrashType.OOM

    def test_oom_by_exit_code_137(self, classifier):
        result = classifier.classify(exit_code=137)
        assert result == CrashType.OOM

    def test_oom_by_exit_code_neg9(self, classifier):
        result = classifier.classify(exit_code=-9)
        assert result == CrashType.OOM

    def test_cuda_error(self, classifier):
        stderr = "CUDA error: an illegal instruction was encountered"
        result = classifier.classify(stderr=stderr, exit_code=1)
        assert result == CrashType.CUDA_ERROR

    def test_cublas_error(self, classifier):
        stderr = "CUBLAS_STATUS_EXECUTION_FAILED"
        result = classifier.classify(stderr=stderr, exit_code=1)
        assert result == CrashType.CUDA_ERROR

    def test_nccl_error(self, classifier):
        stderr = "NCCL error: unhandled system error"
        result = classifier.classify(stderr=stderr)
        assert result == CrashType.CUDA_ERROR

    def test_segfault_exit_code(self, classifier):
        result = classifier.classify(exit_code=-11)
        assert result == CrashType.CUDA_ERROR

    def test_config_invalid_value_error(self, classifier):
        stderr = "ValueError: max_num_batched_tokens must be >= max_num_seqs"
        result = classifier.classify(stderr=stderr)
        assert result == CrashType.CONFIG_INVALID

    def test_config_invalid_unsupported_quant(self, classifier):
        stderr = "quantization method 'squeezellm' not supported for this model"
        result = classifier.classify(stderr=stderr)
        assert result == CrashType.CONFIG_INVALID

    def test_config_invalid_incompatible(self, classifier):
        stderr = "Cannot use chunked prefill with prefix caching"
        result = classifier.classify(stderr=stderr)
        assert result == CrashType.CONFIG_INVALID

    def test_timeout(self, classifier):
        result = classifier.classify(timed_out=True)
        assert result == CrashType.TIMEOUT

    def test_timeout_takes_priority(self, classifier):
        result = classifier.classify(
            stderr="CUDA out of memory", timed_out=True
        )
        assert result == CrashType.TIMEOUT

    def test_startup_failure(self, classifier):
        stderr = "vLLM not found. Install with: pip install vllm"
        result = classifier.classify(stderr=stderr)
        assert result == CrashType.STARTUP_FAILURE

    def test_startup_module_not_found(self, classifier):
        stderr = "ModuleNotFoundError: No module named 'vllm'"
        result = classifier.classify(stderr=stderr)
        assert result == CrashType.STARTUP_FAILURE

    def test_unknown_error(self, classifier):
        stderr = "Something completely unexpected happened"
        result = classifier.classify(stderr=stderr, exit_code=42)
        assert result == CrashType.UNKNOWN

    def test_classify_exception(self, classifier):
        exc = MemoryError("Cannot allocate memory")
        result = classifier.classify_exception(exc)
        assert result == CrashType.OOM

    def test_gpu_memory_utilization_too_high(self, classifier):
        stderr = "ValueError: gpu_memory_utilization 0.99 is too high"
        result = classifier.classify(stderr=stderr)
        assert result == CrashType.OOM
