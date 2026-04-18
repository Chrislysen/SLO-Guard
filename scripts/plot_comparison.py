#!/usr/bin/env python3
"""Compare benchmark runs with publication-quality matplotlib figures.

Two modes:

**Single-seed** (original): take one random-search JSONL and one TBA-TPE
JSONL; produce convergence / phase-transition / config-space / goodput
comparison plots.

    python scripts/plot_comparison.py \\
        --random results/colab_random/random_run.jsonl \\
        --tba-tpe results/tba_tpe/results.jsonl \\
        --output figures/comparison/

**Multi-seed**: take a directory of ``{optimizer}_seed{N}/results.jsonl``
subdirs; produce box plots + per-seed convergence + scatter that reflect
cross-seed variance.

    python scripts/plot_comparison.py \\
        --multiseed results/multiseed/ \\
        --output    figures/multiseed/

JSONL fields used:
  - trial (int)
  - status ("feasible" | "crash")
  - avg_latency_ms (float, only on feasible trials)
  - goodput_tps (float, optional — computed from total_tokens if absent)
  - total_tokens (int, optional)
  - phase (str, optional — TBA-TPE tags "tba-explore"/"tpe-exploit")
  - config (dict with at least max_num_seqs, gpu_memory_utilization, enforce_eager)

Assumption: when goodput_tps is absent, num_requests=5 per trial (the
colab_curl_benchmark default), so goodput = total_tokens / (5 * latency_s).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

NUM_REQUESTS_PER_TRIAL = 5  # curl benchmark default used in these runs

COLOR_RANDOM = "#d95f02"   # orange
COLOR_TBA_TPE = "#1b9e77"  # teal
COLOR_EXPLORE = "#cccccc"  # grey band
COLOR_EXPLOIT = "#a6dba0"  # light green band

_OPT_COLORS = {"random": COLOR_RANDOM, "tba-tpe": COLOR_TBA_TPE}
_OPT_LABELS = {"random": "Random search", "tba-tpe": "TBA-TPE"}
_SEED_DIR_RE = re.compile(r"^(?P<opt>.+)_seed(?P<seed>\d+)$")


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


# ---------------------------------------------------------------------------
# Multi-seed plots
# ---------------------------------------------------------------------------


def _load_compute_multiseed():
    """Load scripts/compute_multiseed_stats.py as a module (not a package)."""
    path = Path(__file__).parent / "compute_multiseed_stats.py"
    spec = importlib.util.spec_from_file_location("compute_multiseed_stats", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compute_multiseed_stats"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_multiseed_trials(
    results_dir: Path,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    """Walk results_dir/{opt}_seed{N}/results.jsonl; return {opt: {seed: trials}}."""
    by_opt: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = _SEED_DIR_RE.match(sub.name)
        if not m:
            continue
        trials_path = sub / "results.jsonl"
        if not trials_path.exists():
            continue
        opt, seed = m.group("opt"), int(m.group("seed"))
        by_opt.setdefault(opt, {})[seed] = load_jsonl(trials_path)
    return by_opt


def _boxplot_two(ax, random_vals, tba_vals, ylabel, title):
    """Shared boxplot styling for the two multi-seed box plots."""
    positions = [1, 2]
    bp = ax.boxplot(
        [random_vals, tba_vals],
        positions=positions, widths=0.5, patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(bp["boxes"], [COLOR_RANDOM, COLOR_TBA_TPE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Overlay individual seed points so the reader sees raw data
    for pos, vals, color in zip(positions, [random_vals, tba_vals],
                                [COLOR_RANDOM, COLOR_TBA_TPE]):
        jitter = [pos + (i - len(vals) / 2) * 0.04 for i in range(len(vals))]
        ax.scatter(jitter, vals, color=color, s=55, zorder=3,
                   edgecolors="white", linewidths=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Random search", "TBA-TPE"])
    _setup_axes(ax, title, "", ylabel)


def plot_multiseed_fastcluster_box(summary, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    r = summary["per_optimizer"]["random"]["fast_cluster_count"]["values"]
    t = summary["per_optimizer"]["tba-tpe"]["fast_cluster_count"]["values"]
    p = summary["mann_whitney"]["fast_cluster_count"]["p"]
    _boxplot_two(
        ax, r, t,
        ylabel="Fast-cluster trials / 15 (higher is better)",
        title=f"Fast-cluster count across 5 seeds  (p={p:.3f})",
    )
    ax.set_ylim(0, 15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_multiseed_posthit_box(summary, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    r = summary["per_optimizer"]["random"]["post_hit_consistency"]["values"]
    t = summary["per_optimizer"]["tba-tpe"]["post_hit_consistency"]["values"]
    p = summary["mann_whitney"]["post_hit_consistency"]["p"]
    _boxplot_two(
        ax, r, t,
        ylabel="Post-hit consistency (higher is better)",
        title=f"Fraction of trials fast after first hit  (p={p:.3f})",
    )
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_multiseed_convergence(by_opt, out_path: Path) -> None:
    """Two-panel best-so-far latency, one line per seed per optimizer."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, opt in zip(axes, ("random", "tba-tpe")):
        color = _OPT_COLORS[opt]
        for _i, (seed, trials) in enumerate(sorted(by_opt[opt].items())):
            trials = sorted(trials, key=lambda r: r["trial"])
            latencies = [t.get("avg_latency_ms") for t in trials]
            best = best_so_far(latencies, better_is_lower=True)
            xs = [t["trial"] for t in trials]
            ax.step(xs, best, where="post", color=color,
                    alpha=0.6, linewidth=1.8, label=f"seed {seed}")
        ax.axhline(500, color="#888", linestyle=":", linewidth=1, alpha=0.7)
        _setup_axes(
            ax, _OPT_LABELS[opt], "Trial",
            "Best-so-far latency (ms)" if opt == "random" else "",
        )
        ax.set_xlim(0.5, 15.5)
        ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)

    fig.suptitle(
        "Best-so-far latency across 5 seeds — lower is better",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multiseed_latency_scatter(summary, out_path: Path) -> None:
    """Dot plot of each seed's best latency, with mean bars overlaid."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    r = summary["per_optimizer"]["random"]["best_latency_ms"]
    t = summary["per_optimizer"]["tba-tpe"]["best_latency_ms"]
    p = summary["mann_whitney"]["best_latency_ms"]["p"]

    for pos, agg, color in [(1, r, COLOR_RANDOM), (2, t, COLOR_TBA_TPE)]:
        vals = agg["values"]
        jitter = [pos + (i - len(vals) / 2) * 0.05 for i in range(len(vals))]
        ax.scatter(jitter, vals, color=color, s=90, zorder=3,
                   edgecolors="white", linewidths=1.2)
        # Mean bar
        ax.hlines(agg["mean"], pos - 0.2, pos + 0.2,
                  colors="black", linewidth=2.5, zorder=4)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Random search", "TBA-TPE"])
    _setup_axes(
        ax,
        f"Best latency per seed  (p={p:.2f}, tied)",
        "",
        "Best avg latency (ms)",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _single_seed(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    random_trials = load_jsonl(args.random)
    tba_tpe_trials = load_jsonl(args.tba_tpe)
    print(f"Loaded {len(random_trials)} random trials, {len(tba_tpe_trials)} TBA-TPE trials")
    plot_convergence(random_trials, tba_tpe_trials, args.output / "convergence.png")
    plot_phase_transition(tba_tpe_trials, args.output / "phase_transition.png")
    plot_config_space(random_trials, tba_tpe_trials, args.output / "config_space.png")
    plot_goodput(random_trials, tba_tpe_trials, args.output / "goodput_comparison.png")
    print(f"Wrote 4 figures to {args.output}/")


def _multi_seed(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    compute_mod = _load_compute_multiseed()
    summary = compute_mod.compute(args.multiseed, fast_threshold_ms=args.fast_threshold_ms)
    by_opt = _load_multiseed_trials(args.multiseed)
    n_seeds = len(by_opt.get("random", {}))
    print(f"Loaded {n_seeds} seeds × 2 optimizers from {args.multiseed}")
    plot_multiseed_fastcluster_box(summary, args.output / "multiseed_fastcluster_box.png")
    plot_multiseed_posthit_box(summary, args.output / "multiseed_posthit_box.png")
    plot_multiseed_convergence(by_opt, args.output / "multiseed_convergence.png")
    plot_multiseed_latency_scatter(summary, args.output / "multiseed_latency_scatter.png")
    print(f"Wrote 4 figures to {args.output}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--multiseed", type=Path, metavar="DIR",
        help="Multi-seed mode: directory of {optimizer}_seed{N}/results.jsonl",
    )
    mode.add_argument(
        "--random", type=Path, metavar="PATH",
        help="Single-seed mode: random-search JSONL (requires --tba-tpe)",
    )
    parser.add_argument(
        "--tba-tpe", type=Path, metavar="PATH",
        help="Single-seed mode: TBA-TPE JSONL",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: figures/comparison or figures/multiseed)",
    )
    parser.add_argument(
        "--fast-threshold-ms", type=float, default=1000.0,
        help="Latency below this counts as fast-cluster (multi-seed only)",
    )
    args = parser.parse_args()

    if args.multiseed is not None:
        args.output = args.output or Path("figures/multiseed")
        _multi_seed(args)
    else:
        if args.tba_tpe is None:
            parser.error("--random requires --tba-tpe")
        args.output = args.output or Path("figures/comparison")
        _single_seed(args)


if __name__ == "__main__":
    main()
