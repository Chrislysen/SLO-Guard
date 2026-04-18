"""Async load generator for LLM serving benchmarks.

Three modes:
  - FixedRate: Poisson arrivals at target req/s
  - Burst: step-function bursts (baseline -> peak -> baseline)
  - TraceReplay: replay inter-arrival times from CSV trace files

Sends requests to vLLM's OpenAI-compatible /v1/chat/completions with
streaming enabled, collecting per-token timestamps for TTFT/ITL metrics.
"""
from __future__ import annotations

import asyncio
import csv
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from sloguard.types import TimeoutConfig

logger = logging.getLogger(__name__)


@dataclass
class RequestResult:
    """Per-request result with timing information."""

    request_id: int
    prompt_tokens: int
    output_tokens: int
    send_time: float  # monotonic time when request was sent
    first_token_time: float | None = None  # time of first output token
    token_times: list[float] = field(default_factory=list)  # all token arrival times
    end_time: float | None = None
    error: str | None = None
    success: bool = True

    @property
    def ttft_ms(self) -> float | None:
        """Time to first token in milliseconds."""
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.send_time) * 1000

    @property
    def itl_ms_list(self) -> list[float]:
        """Inter-token latencies in milliseconds."""
        if len(self.token_times) < 2:
            return []
        return [
            (self.token_times[i] - self.token_times[i - 1]) * 1000
            for i in range(1, len(self.token_times))
        ]

    @property
    def total_latency_ms(self) -> float | None:
        """Total request latency in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.send_time) * 1000


@dataclass
class WorkloadConfig:
    """Configuration for a load generation run."""

    request_rate: float  # requests per second
    num_requests: int = 100
    prompt_len_min: int = 128
    prompt_len_max: int = 512
    output_len_min: int = 64
    output_len_max: int = 256
    model: str = ""  # model name for the API
    timeout_per_request: float = 30.0  # seconds


class LoadGenerator:
    """Base class for load generation."""

    def __init__(
        self,
        base_url: str,
        workload: WorkloadConfig,
        seed: int = 42,
        timeouts: TimeoutConfig | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.workload = workload
        self.rng = random.Random(seed)
        self.results: list[RequestResult] = []
        self.timeouts = timeouts or TimeoutConfig()

    async def run(self, trial_timeout: float | None = None) -> list[RequestResult]:
        """Run the load generation and return per-request results.

        Args:
            trial_timeout: Hard timeout for the entire trial in seconds.
                Defaults to the runner's TimeoutConfig.per_trial_s. If the
                cap is exceeded, returns whatever results were collected.
        """
        cap = trial_timeout if trial_timeout is not None else self.timeouts.per_trial_s
        try:
            return await asyncio.wait_for(self._run_inner(), timeout=cap)
        except asyncio.TimeoutError:
            logger.warning(
                "Trial timeout after %.0fs — returning %d results collected so far",
                cap, len(self.results),
            )
            return self.results

    async def _run_inner(self) -> list[RequestResult]:
        """Inner run loop — called within the trial timeout."""
        self.results = []
        inter_arrival_times = self._generate_inter_arrival_times()

        connector = aiohttp.TCPConnector(limit=self.workload.num_requests)
        timeout = aiohttp.ClientTimeout(
            total=self.workload.timeout_per_request,
            sock_read=30,  # kill stuck streaming reads after 30s of silence
        )
        per_req_cap = self.timeouts.per_request_s
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for i in range(self.workload.num_requests):
                if i > 0 and i - 1 < len(inter_arrival_times):
                    await asyncio.sleep(inter_arrival_times[i - 1])

                prompt_len = self.rng.randint(
                    self.workload.prompt_len_min, self.workload.prompt_len_max
                )
                output_len = self.rng.randint(
                    self.workload.output_len_min, self.workload.output_len_max
                )

                try:
                    result = await asyncio.wait_for(
                        self._send_request(session, i, prompt_len, output_len),
                        timeout=per_req_cap,
                    )
                except asyncio.TimeoutError:
                    result = RequestResult(
                        request_id=i,
                        prompt_tokens=prompt_len,
                        output_tokens=0,
                        send_time=time.monotonic(),
                        end_time=time.monotonic(),
                        error=f"Hard timeout after {per_req_cap:.0f}s",
                        success=False,
                    )

                self.results.append(result)

                # If 3+ consecutive failures, server is likely broken — abort early
                if len(self.results) >= 3:
                    recent = self.results[-3:]
                    if all(not r.success for r in recent):
                        logger.warning(
                            "3 consecutive request failures — aborting load generation"
                        )
                        break

        return self.results

    async def _send_request(
        self,
        session: aiohttp.ClientSession,
        request_id: int,
        prompt_len: int,
        output_len: int,
    ) -> RequestResult:
        """Send a single streaming request and collect token timestamps."""
        prompt = self._make_prompt(prompt_len)

        payload = {
            "model": self.workload.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_len,
            "stream": True,
            "temperature": 0.0,
        }

        result = RequestResult(
            request_id=request_id,
            prompt_tokens=prompt_len,
            output_tokens=0,
            send_time=time.monotonic(),
        )

        try:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    result.error = f"HTTP {resp.status}: {body[:500]}"
                    result.success = False
                    result.end_time = time.monotonic()
                    return result

                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()
                    if not decoded.startswith("data: "):
                        continue
                    data = decoded[6:]
                    if data == "[DONE]":
                        break

                    now = time.monotonic()
                    result.token_times.append(now)
                    result.output_tokens += 1

                    if result.first_token_time is None:
                        result.first_token_time = now

        except asyncio.TimeoutError:
            result.error = "Request timed out"
            result.success = False
        except aiohttp.ClientError as e:
            result.error = f"Client error: {e}"
            result.success = False
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            result.success = False

        result.end_time = time.monotonic()
        return result

    def _make_prompt(self, target_tokens: int) -> str:
        """Generate a prompt of approximately target_tokens length.

        Uses repeated words to approximate token count (~0.75 tokens per word).
        """
        words_needed = int(target_tokens / 0.75)
        vocabulary = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "and", "then", "runs", "through", "forest", "into", "river",
            "under", "bright", "blue", "sky", "with", "warm", "gentle", "breeze",
            "across", "green", "meadow", "between", "tall", "ancient", "trees",
        ]
        words = [self.rng.choice(vocabulary) for _ in range(words_needed)]
        return "Summarize the following text: " + " ".join(words)

    def _generate_inter_arrival_times(self) -> list[float]:
        """Generate inter-arrival times. Override in subclasses."""
        raise NotImplementedError


class FixedRateGenerator(LoadGenerator):
    """Poisson arrivals at a fixed target rate."""

    def _generate_inter_arrival_times(self) -> list[float]:
        if self.workload.request_rate <= 0:
            return [0.0] * self.workload.num_requests
        mean_interval = 1.0 / self.workload.request_rate
        return [
            self.rng.expovariate(1.0 / mean_interval)
            for _ in range(self.workload.num_requests - 1)
        ]


class BurstGenerator(LoadGenerator):
    """Step-function burst pattern: baseline -> peak -> baseline.

    Args:
        baseline_rate: Normal request rate (req/s)
        peak_rate: Burst request rate (req/s)
        burst_start_frac: When burst starts (fraction of total requests)
        burst_end_frac: When burst ends (fraction of total requests)
    """

    def __init__(
        self,
        base_url: str,
        workload: WorkloadConfig,
        seed: int = 42,
        timeouts: TimeoutConfig | None = None,
        baseline_rate: float | None = None,
        peak_rate: float | None = None,
        burst_start_frac: float = 0.3,
        burst_end_frac: float = 0.7,
    ):
        super().__init__(base_url, workload, seed, timeouts)
        self.baseline_rate = baseline_rate or workload.request_rate
        self.peak_rate = peak_rate or workload.request_rate * 5.0
        self.burst_start_frac = burst_start_frac
        self.burst_end_frac = burst_end_frac

    def _generate_inter_arrival_times(self) -> list[float]:
        n = self.workload.num_requests - 1
        burst_start = int(n * self.burst_start_frac)
        burst_end = int(n * self.burst_end_frac)
        times = []

        for i in range(n):
            if burst_start <= i < burst_end:
                rate = self.peak_rate
            else:
                rate = self.baseline_rate

            if rate <= 0:
                times.append(0.0)
            else:
                times.append(self.rng.expovariate(rate))

        return times


class TraceReplayGenerator(LoadGenerator):
    """Replay inter-arrival times from a CSV trace file.

    Expected CSV format: one column of inter-arrival times in seconds,
    or a column named 'inter_arrival_s'.

    If the trace is shorter than num_requests, it wraps around.
    If longer, it truncates.
    """

    def __init__(
        self,
        base_url: str,
        workload: WorkloadConfig,
        trace_path: str | Path,
        seed: int = 42,
        timeouts: TimeoutConfig | None = None,
        time_column: str = "inter_arrival_s",
        scale_factor: float = 1.0,
    ):
        super().__init__(base_url, workload, seed, timeouts)
        self.trace_path = Path(trace_path)
        self.time_column = time_column
        self.scale_factor = scale_factor
        self._trace_times: list[float] = []
        self._load_trace()

    def _load_trace(self) -> None:
        """Load inter-arrival times from CSV."""
        if not self.trace_path.exists():
            logger.warning("Trace file not found: %s, using Poisson fallback", self.trace_path)
            return

        with open(self.trace_path) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and self.time_column in reader.fieldnames:
                for row in reader:
                    try:
                        t = float(row[self.time_column]) * self.scale_factor
                        self._trace_times.append(max(0.0, t))
                    except (ValueError, KeyError):
                        continue
            else:
                # Try first column as plain float
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        t = float(line.split(",")[0]) * self.scale_factor
                        self._trace_times.append(max(0.0, t))
                    except ValueError:
                        continue

        logger.info("Loaded %d inter-arrival times from trace", len(self._trace_times))

    def _generate_inter_arrival_times(self) -> list[float]:
        n = self.workload.num_requests - 1
        if not self._trace_times:
            # Fallback to Poisson
            if self.workload.request_rate <= 0:
                return [0.0] * n
            mean_interval = 1.0 / self.workload.request_rate
            return [self.rng.expovariate(1.0 / mean_interval) for _ in range(n)]

        # Wrap trace if needed
        times = []
        for i in range(n):
            idx = i % len(self._trace_times)
            times.append(self._trace_times[idx])
        return times


def create_generator(
    mode: str,
    base_url: str,
    workload: WorkloadConfig,
    seed: int = 42,
    timeouts: TimeoutConfig | None = None,
    **kwargs: Any,
) -> LoadGenerator:
    """Factory for load generators.

    Args:
        mode: "fixed", "burst", or "trace"
        base_url: vLLM server URL
        workload: Workload configuration
        seed: Random seed
        timeouts: Optional TimeoutConfig — defaults to TimeoutConfig() inside
            each generator if not supplied
        **kwargs: Extra args for specific generator types
    """
    if mode == "fixed":
        return FixedRateGenerator(base_url, workload, seed, timeouts)
    elif mode == "burst":
        return BurstGenerator(base_url, workload, seed, timeouts, **kwargs)
    elif mode == "trace":
        trace_path = kwargs.pop("trace_path", "trace.csv")
        return TraceReplayGenerator(
            base_url, workload, trace_path, seed, timeouts, **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown load generator mode: {mode!r}. Use 'fixed', 'burst', or 'trace'",
        )
