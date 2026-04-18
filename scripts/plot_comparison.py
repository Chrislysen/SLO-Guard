#!/usr/bin/env python3
"""Compare two benchmark runs (random vs TBA-TPE) with four figures.

Inputs: two JSONL files with per-trial records. Fields used:
  - trial (int)
  - status ("feasible" | "crash")
  - avg_latency_ms (float, only on feasible trials)
  - goodput_tps (float, optional — computed from total_tokens if absent)
  - total_tokens (int, optional — used when goodput_tps is missing)
  - phase (str, optional — TBA-TPE tags trials with "tba-explore"/"tpe-exploit")
  - config (dict with at least max_num_seqs, gpu_memory_utilization, enforce_eager)

Assumption: when goodput_tps is absent, the curl benchmark ran with
num_requests=5 (confirmed against scripts/colab_curl_benchmark.py
defaults), so goodput = total_tokens / (5 * avg_latency_ms/1000).

Usage:
    python scripts/plot_comparison.py \\
        --random results/colab_random/random_run.jsonl \\
        --tba-tpe results/tba_tpe/results.jsonl \\
        --output figures/comparison/
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

NUM_REQUESTS_PER_TRIAL = 5  # curl benchmark default used in these runs

COLOR_RANDOM = "#d95f02"   # orange
COLOR_TBA_TPE = "#1b9e77"  # teal
COLOR_EXPLORE = "#cccccc"  # grey band
COLOR_EXPLOIT = "#a6dba0"  # light green band


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def goodput_for(trial: dict[str, Any]) -> float | None:
    """Return goodput (tokens/s) from the trial record, or None on crash."""
    if trial.get("status") != "feasible":
        return None
    if "goodput_tps" in trial:
        return float(trial["goodput_tps"])
    total = trial.get("total_tokens")
    latency = trial.get("avg_latency_ms")
    if total is None or latency is None or latency <= 0:
        return None
    return total / (NUM_REQUESTS_PER_TRIAL * latency / 1000.0)


def best_so_far(series: list[float | None], better_is_lower: bool) -> list[float | None]:
    """Cumulative best-so-far ignoring None entries."""
    best: float | None = None
    out: list[float | None] = []
    for x in series:
        if x is not None:
            if best is None or (x < best if better_is_lower else x > best):
                best = x
        out.append(best)
    return out


def _setup_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_convergence(random_trials, tba_tpe_trials, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for trials, label, color in [
        (random_trials, "Random search", COLOR_RANDOM),
        (tba_tpe_trials, "TBA-TPE", COLOR_TBA_TPE),
    ]:
        latencies = [t.get("avg_latency_ms") for t in trials]
        best = best_so_far(latencies, better_is_lower=True)
        xs = [t["trial"] for t in trials]
        ax.step(xs, best, where="post", label=label, color=color, linewidth=2.2)
        # Mark trials with actual measurements (not just the step carry-over)
        for x, lat in zip(xs, latencies):
            if lat is not None:
                ax.plot(x, lat, "o", color=color, alpha=0.35, markersize=4)

    ax.axhline(500, color="#888", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(0.5, 520, "fast cluster (~440ms)", fontsize=9, color="#555")
    _setup_axes(
        ax,
        "Best-so-far latency over trials",
        "Trial",
        "Latency (ms) — lower is better",
    )
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0.5, max(len(random_trials), len(tba_tpe_trials)) + 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_phase_transition(tba_tpe_trials, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    xs = [t["trial"] for t in tba_tpe_trials]
    ys = [t.get("avg_latency_ms") for t in tba_tpe_trials]
    phases = [t.get("phase", "") for t in tba_tpe_trials]

    # Find handoff boundary: last explore trial + 0.5
    explore_trials = [x for x, ph in zip(xs, phases) if "explore" in ph]
    exploit_trials = [x for x, ph in zip(xs, phases) if "exploit" in ph]
    handoff = (max(explore_trials) + min(exploit_trials)) / 2 if exploit_trials else None

    if handoff is not None:
        ax.axvspan(0, handoff, color=COLOR_EXPLORE, alpha=0.5, label="TBA explore")
        ax.axvspan(handoff, max(xs) + 0.5, color=COLOR_EXPLOIT, alpha=0.5, label="TPE exploit")
        ax.axvline(handoff, color="#333", linestyle="--", linewidth=1.5)
        ax.text(
            handoff, max(y for y in ys if y) * 0.97,
            f" handoff @ trial {int(handoff + 0.5)}",
            fontsize=10, color="#333",
        )

    for x, y, ph in zip(xs, ys, phases):
        if y is None:
            continue
        marker = "s" if "explore" in ph else "o"
        color = "#555" if "explore" in ph else COLOR_TBA_TPE
        ax.plot(x, y, marker=marker, color=color, markersize=10, markeredgecolor="white")

    _setup_axes(
        ax,
        "TBA-TPE: explore → exploit phase transition",
        "Trial",
        "Latency (ms)",
    )
    ax.legend(loc="center right", frameon=False)
    ax.set_xlim(0.5, max(xs) + 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_config_space(random_trials, tba_tpe_trials, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    def scatter(trials, color, label):
        for t in trials:
            cfg = t.get("config", {})
            lat = t.get("avg_latency_ms")
            if lat is None:
                continue
            seqs = cfg.get("max_num_seqs")
            util = cfg.get("gpu_memory_utilization")
            eager = cfg.get("enforce_eager", False)
            if seqs is None or util is None:
                continue
            # Size scales with latency so fast configs are small dots, slow ones large.
            size = 40 + (lat / 20)
            marker = "s" if eager else "o"
            ax.scatter(
                seqs, util, s=size, marker=marker, c=color,
                alpha=0.7, edgecolors="white", linewidths=0.8,
            )
        # Invisible point for legend
        ax.scatter([], [], c=color, label=label, s=80)

    scatter(random_trials, COLOR_RANDOM, "Random search")
    scatter(tba_tpe_trials, COLOR_TBA_TPE, "TBA-TPE")

    # Shape legend
    ax.scatter([], [], marker="s", c="#555", s=80, label="enforce_eager=True (slow)")
    ax.scatter([], [], marker="o", c="#555", s=80, label="enforce_eager=False (fast)")

    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    _setup_axes(
        ax,
        "Configuration space — marker size ∝ latency",
        "max_num_seqs (log₂)",
        "gpu_memory_utilization",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_goodput(random_trials, tba_tpe_trials, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for trials, label, color in [
        (random_trials, "Random search", COLOR_RANDOM),
        (tba_tpe_trials, "TBA-TPE", COLOR_TBA_TPE),
    ]:
        xs = [t["trial"] for t in trials]
        gp = [goodput_for(t) for t in trials]
        # Plot only measured points; don't connect across crashes
        valid_xs = [x for x, g in zip(xs, gp) if g is not None]
        valid_gp = [g for g in gp if g is not None]
        ax.plot(valid_xs, valid_gp, "o-", color=color, label=label, linewidth=2, markersize=6)
        # Mark crashes with an X
        crash_xs = [x for x, g in zip(xs, gp) if g is None]
        for cx in crash_xs:
            ax.plot(cx, 0, "x", color=color, markersize=12, markeredgewidth=2.5)

    _setup_axes(
        ax,
        "Goodput per trial (crashes shown at y=0 with ×)",
        "Trial",
        "Goodput (tokens/s) — higher is better",
    )
    ax.legend(loc="center right", frameon=False)
    ax.set_xlim(0.5, max(len(random_trials), len(tba_tpe_trials)) + 0.5)
    ax.set_ylim(bottom=-10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random", type=Path, required=True, help="Random-search JSONL",
    )
    parser.add_argument(
        "--tba-tpe", type=Path, required=True, help="TBA-TPE JSONL",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("figures/comparison"),
        help="Output directory for PNGs",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    random_trials = load_jsonl(args.random)
    tba_tpe_trials = load_jsonl(args.tba_tpe)

    print(f"Loaded {len(random_trials)} random trials, {len(tba_tpe_trials)} TBA-TPE trials")

    plot_convergence(random_trials, tba_tpe_trials, args.output / "convergence.png")
    plot_phase_transition(tba_tpe_trials, args.output / "phase_transition.png")
    plot_config_space(random_trials, tba_tpe_trials, args.output / "config_space.png")
    plot_goodput(random_trials, tba_tpe_trials, args.output / "goodput_comparison.png")

    print(f"Wrote 4 figures to {args.output}/")


if __name__ == "__main__":
    main()
