"""vLLM server lifecycle management.

Starts vLLM as a subprocess, polls for readiness, captures stderr for
crash classification, and handles graceful + forced shutdown.
"""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# How long to wait for the server to start (seconds)
DEFAULT_STARTUP_TIMEOUT = 300
# How long to wait between health checks (seconds)
HEALTH_CHECK_INTERVAL = 2
# How long to wait for graceful shutdown (seconds)
SHUTDOWN_TIMEOUT = 15


class VLLMServerManager:
    """Manages the lifecycle of a vLLM OpenAI-compatible API server.

    Usage:
        manager = VLLMServerManager(model="Qwen/Qwen2-1.5B", port=8000)
        manager.start(config={"gpu_memory_utilization": 0.8, ...})
        # ... run benchmarks ...
        manager.stop()

    Or as context manager:
        with VLLMServerManager(model="Qwen/Qwen2-1.5B") as manager:
            manager.start(config={...})
            # ... run benchmarks ...
    """

    def __init__(
        self,
        model: str,
        port: int = 8000,
        host: str = "0.0.0.0",
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    ):
        self.model = model
        self.port = port
        self.host = host
        self.startup_timeout = startup_timeout
        self._process: subprocess.Popen | None = None
        self._stderr_output: str = ""
        self._startup_time: float = 0.0

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def stderr_output(self) -> str:
        return self._stderr_output

    @property
    def startup_time(self) -> float:
        return self._startup_time

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, config: dict[str, Any]) -> bool:
        """Start vLLM server with the given configuration.

        Returns True if server started successfully, False otherwise.
        Captures stderr for crash classification on failure.
        """
        if self.is_running:
            self.stop()

        cmd = self._build_command(config)
        logger.info("Starting vLLM: %s", " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            self._stderr_output = "vLLM not found. Install with: pip install vllm"
            logger.error(self._stderr_output)
            return False
        except Exception as e:
            self._stderr_output = f"Failed to start vLLM process: {e}"
            logger.error(self._stderr_output)
            return False

        start_time = time.monotonic()
        success = self._wait_for_ready(start_time)
        self._startup_time = time.monotonic() - start_time

        if not success:
            self._capture_stderr()
            self.stop()
            return False

        logger.info("vLLM server ready in %.1fs", self._startup_time)
        return True

    def stop(self) -> None:
        """Stop the vLLM server gracefully, force-kill if needed."""
        if self._process is None:
            return

        self._capture_stderr()

        if self._process.poll() is None:
            logger.info("Stopping vLLM server (pid=%d)", self._process.pid)
            self._process.terminate()
            try:
                self._process.wait(timeout=SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                logger.warning("Force-killing vLLM server")
                self._process.kill()
                self._process.wait(timeout=5)

        self._process = None

    def health_check(self) -> bool:
        """Check if the vLLM server is responding."""
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(f"{self.base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _build_command(self, config: dict[str, Any]) -> list[str]:
        """Build the vLLM CLI command from a config dict."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
        ]

        # Map config knobs to vLLM CLI args
        knob_to_flag = {
            "quantization": "--quantization",
            "max_num_seqs": "--max-num-seqs",
            "max_num_batched_tokens": "--max-num-batched-tokens",
            "gpu_memory_utilization": "--gpu-memory-utilization",
            "enforce_eager": "--enforce-eager",
            "enable_chunked_prefill": "--enable-chunked-prefill",
            "enable_prefix_caching": "--enable-prefix-caching",
            "block_size": "--block-size",
            "swap_space": "--swap-space",
            "max_model_len": "--max-model-len",
            "dtype": "--dtype",
        }

        for knob, flag in knob_to_flag.items():
            if knob not in config:
                continue
            value = config[knob]

            # Skip default quantization
            if knob == "quantization" and value == "fp16":
                continue

            # Boolean flags (no value needed if True, skip if False)
            if knob in ("enforce_eager", "enable_chunked_prefill", "enable_prefix_caching"):
                if value is True:
                    cmd.append(flag)
                continue

            cmd.extend([flag, str(value)])

        return cmd

    def _wait_for_ready(self, start_time: float) -> bool:
        """Poll health endpoint until server is ready or timeout."""
        while time.monotonic() - start_time < self.startup_timeout:
            # Check if process died
            if self._process.poll() is not None:
                logger.error("vLLM process exited with code %d", self._process.returncode)
                return False

            if self.health_check():
                return True

            time.sleep(HEALTH_CHECK_INTERVAL)

        logger.error("vLLM startup timed out after %.0fs", self.startup_timeout)
        return False

    def _capture_stderr(self) -> None:
        """Read any available stderr from the process."""
        if self._process is None or self._process.stderr is None:
            return
        try:
            # Non-blocking read of whatever stderr has
            import select
            import os

            if hasattr(select, "select"):
                # Unix-like: use select for non-blocking check
                ready, _, _ = select.select([self._process.stderr], [], [], 0.1)
                if ready:
                    data = self._process.stderr.read()
                    if data:
                        self._stderr_output += data
            else:
                # Windows fallback: try to read if process is done
                if self._process.poll() is not None:
                    data = self._process.stderr.read()
                    if data:
                        self._stderr_output += data
        except Exception:
            pass

    def __enter__(self) -> VLLMServerManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
