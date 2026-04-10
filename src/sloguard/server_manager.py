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
DEFAULT_STARTUP_TIMEOUT = 120
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

        # Use offline mode so vLLM doesn't hit HuggingFace on every restart
        env = {**__import__("os").environ, "HF_HUB_OFFLINE": "1"}

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
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
        """Build the vLLM 0.19 CLI command from a config dict.

        Only emits flags for knobs present in the config. Skips defaults
        so vLLM can auto-detect where appropriate.
        """
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
        ]

        # Value flags: knob -> (CLI flag, skip_value)
        # skip_value: if config value equals this, don't emit the flag
        # NOTE: dtype and block_size intentionally omitted — let vLLM
        # auto-detect dtype (A100 prefers bfloat16) and use default block_size.
        value_flags: dict[str, tuple[str, Any]] = {
            "quantization": ("--quantization", "fp16"),
            "max_num_seqs": ("--max-num-seqs", None),
            "max_num_batched_tokens": ("--max-num-batched-tokens", None),
            "gpu_memory_utilization": ("--gpu-memory-utilization", None),
            "max_model_len": ("--max-model-len", None),
        }

        # Boolean flags with --no- variants (BooleanOptionalAction in vLLM)
        bool_flags = {
            "enable_chunked_prefill": ("--enable-chunked-prefill", "--no-enable-chunked-prefill"),
        }

        # store_true flags — only emit when True, omit when False
        # (lets vLLM use its defaults; avoids --no-* variants that may
        # not exist in all vLLM 0.19 builds).
        true_only_flags = {
            "enforce_eager": "--enforce-eager",
            "enable_prefix_caching": "--enable-prefix-caching",
        }

        for knob, (flag, skip_val) in value_flags.items():
            if knob not in config:
                continue
            value = config[knob]
            if skip_val is not None and value == skip_val:
                continue
            cmd.extend([flag, str(value)])

        for knob, flag in true_only_flags.items():
            if config.get(knob):
                cmd.append(flag)

        for knob, (true_flag, false_flag) in bool_flags.items():
            if knob not in config:
                continue
            cmd.append(true_flag if config[knob] else false_flag)

        return cmd

    def _wait_for_ready(self, start_time: float) -> bool:
        """Poll health endpoint until server is ready or timeout."""
        while time.monotonic() - start_time < self.startup_timeout:
            # Check if process died
            if self._process.poll() is not None:
                # Read all stderr before reporting
                self._capture_stderr_blocking()
                logger.error(
                    "vLLM process exited with code %d:\n%s",
                    self._process.returncode,
                    self._stderr_output if self._stderr_output else "(no stderr)",
                )
                return False

            if self.health_check():
                return True

            time.sleep(HEALTH_CHECK_INTERVAL)

        logger.error("vLLM startup timed out after %.0fs", self.startup_timeout)
        return False

    def _capture_stderr(self) -> None:
        """Read any available stderr from the process (non-blocking)."""
        if self._process is None or self._process.stderr is None:
            return
        try:
            if self._process.poll() is not None:
                self._capture_stderr_blocking()
            else:
                import select
                if hasattr(select, "select"):
                    ready, _, _ = select.select([self._process.stderr], [], [], 0.1)
                    if ready:
                        data = self._process.stderr.read()
                        if data:
                            self._stderr_output += data
        except Exception:
            pass

    def _capture_stderr_blocking(self) -> None:
        """Read all remaining stderr (use only when process has exited)."""
        if self._process is None or self._process.stderr is None:
            return
        try:
            data = self._process.stderr.read()
            if data:
                self._stderr_output += data
        except Exception:
            pass

    def __enter__(self) -> VLLMServerManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
