"""Tests for LoadGenerator — concurrency behavior and circuit breaker.

Uses aiohttp.test_utils to stand up a local HTTP server that mimics the
OpenAI chat-completions streaming protocol with configurable delays and
failure rates. No GPU, no vLLM.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import pytest
from aiohttp import web

from sloguard.load_generator import FixedRateGenerator, WorkloadConfig
from sloguard.types import TimeoutConfig

pytestmark = pytest.mark.asyncio


@dataclass
class _FakeServerConfig:
    delay_s: float = 0.0
    fail_every_n: int = 0      # 0 = never fail
    fail_first_n: int = 0      # fail the first N requests only
    fail_probability: float = 0.0  # 0..1


def _build_fake_app(cfg: _FakeServerConfig) -> web.Application:
    """Fake /v1/chat/completions that streams a tiny OpenAI-format response."""
    state = {"count": 0}

    async def handler(request: web.Request) -> web.StreamResponse:
        state["count"] += 1
        i = state["count"]

        should_fail = False
        if cfg.fail_first_n and i <= cfg.fail_first_n:
            should_fail = True
        if cfg.fail_every_n and i % cfg.fail_every_n == 0:
            should_fail = True
        if cfg.fail_probability > 0:
            import random as _random
            if _random.random() < cfg.fail_probability:
                should_fail = True

        if cfg.delay_s > 0:
            await asyncio.sleep(cfg.delay_s)

        if should_fail:
            return web.Response(status=500, text='{"error": "synthetic failure"}')

        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)
        for token in ("hello", "world"):
            chunk = (
                'data: {"choices":[{"delta":{"content":"' + token + '"}}]}\n\n'
            )
            await resp.write(chunk.encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    return app


@pytest.fixture
async def fake_server(aiohttp_server):
    # aiohttp_server is provided by pytest-aiohttp; we instead use a plain
    # local loop because pytest-aiohttp isn't in the dev deps. Spin our own.
    ...


async def _run_with_server(cfg: _FakeServerConfig, workload: WorkloadConfig):
    """Spin a local TCPSite on an ephemeral port, run the workload, tear it down."""
    app = _build_fake_app(cfg)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    # TCPSite doesn't expose the bound port directly; ask the server.
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    base_url = f"http://127.0.0.1:{port}"
    try:
        gen = FixedRateGenerator(
            base_url=base_url, workload=workload, seed=42,
            timeouts=TimeoutConfig(per_request_s=10.0, per_trial_s=30.0),
        )
        t0 = time.monotonic()
        results = await gen.run(trial_timeout=30.0)
        wall_clock = time.monotonic() - t0
        return gen, results, wall_clock
    finally:
        await runner.cleanup()


async def test_concurrent_load_overlaps_requests():
    """10 requests at 10 req/s with a 100ms-per-request endpoint should
    finish in ~1s (concurrent), not ~2s (serial).

    The exact concurrent wall-clock is arrival_span + service_time ≈
    0.9 + 0.1 ≈ 1.0s; the serial bound is sum(service) = 1.0s, which at
    10 req/s barely differs. Use 1.5s as a ceiling to fail cleanly for
    any regression to pre-concurrent behavior (which ran 1.9s+ on CI).
    Peak-concurrency uses a loose >= 2 bound — at rate × service = 1.0
    the steady-state in-flight is only 1, and Poisson arrivals alone
    push it to 2-4 in practice. A stricter bound is in
    test_peak_concurrency_respects_max.
    """
    cfg = _FakeServerConfig(delay_s=0.1)
    workload = WorkloadConfig(
        request_rate=10.0, num_requests=10,
        prompt_len_min=4, prompt_len_max=4,
        output_len_min=2, output_len_max=2,
        model="fake", max_concurrency=20,
    )
    gen, results, wall_clock = await _run_with_server(cfg, workload)

    assert len(results) == 10
    assert all(r.success for r in results)
    assert wall_clock < 1.5, f"expected concurrent run < 1.5s, got {wall_clock:.2f}s"
    assert gen.peak_concurrency >= 2, (
        f"peak_concurrency={gen.peak_concurrency}, expected >= 2 (serial would be 1)"
    )


async def test_serial_wall_clock_is_much_slower_baseline():
    """With a service time larger than the inter-arrival, max_concurrency=1
    makes the run back up into a queue. Picks 20 req/s × 200ms service so
    the concurrent baseline (~0.6s) and the serial baseline (~2.0s) are
    separated by > 1s — safe margin on any CI scheduler."""
    cfg = _FakeServerConfig(delay_s=0.2)
    workload = WorkloadConfig(
        request_rate=20.0, num_requests=10,
        prompt_len_min=4, prompt_len_max=4,
        output_len_min=2, output_len_max=2,
        model="fake", max_concurrency=1,
    )
    gen, results, wall_clock = await _run_with_server(cfg, workload)
    assert len(results) == 10
    assert gen.peak_concurrency == 1
    # Serial: 10 × 0.2s = 2.0s. Concurrent baseline is ~0.6s.
    assert wall_clock > 1.5, (
        f"expected serialized run >= 1.5s, got {wall_clock:.2f}s"
    )


async def test_circuit_breaker_aborts_on_high_failure_rate():
    """Endpoint fails 80% of the time. Circuit breaker should abort once
    we have >=5 completions and failure_rate > 0.5."""
    cfg = _FakeServerConfig(delay_s=0.01, fail_probability=0.8)
    workload = WorkloadConfig(
        request_rate=20.0, num_requests=20,
        prompt_len_min=4, prompt_len_max=4,
        output_len_min=2, output_len_max=2,
        model="fake", max_concurrency=5,
    )
    _, results, _ = await _run_with_server(cfg, workload)

    # At least one request must have been aborted by the circuit breaker.
    aborted = [r for r in results if r.error and "circuit breaker" in r.error.lower()]
    completed = [r for r in results if not (r.error and "circuit breaker" in r.error.lower())]
    assert len(aborted) > 0, "circuit breaker should have aborted remaining requests"
    # And the breaker should fire only after some completions —
    # we shouldn't give up before trying at least a handful.
    assert len(completed) >= 5, (
        f"breaker fired too early: only {len(completed)} completions"
    )


async def test_peak_concurrency_respects_max():
    """Never exceeds max_concurrency even with high arrival rate."""
    cfg = _FakeServerConfig(delay_s=0.2)
    workload = WorkloadConfig(
        request_rate=50.0, num_requests=20,
        prompt_len_min=4, prompt_len_max=4,
        output_len_min=2, output_len_max=2,
        model="fake", max_concurrency=3,
    )
    gen, results, _ = await _run_with_server(cfg, workload)
    assert gen.peak_concurrency <= 3, (
        f"peak_concurrency={gen.peak_concurrency} exceeded max=3"
    )
    # But it should actually *hit* the cap, not just stay at 1.
    assert gen.peak_concurrency >= 2
