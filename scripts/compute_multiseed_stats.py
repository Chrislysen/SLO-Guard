#!/usr/bin/env python3
"""Compute aggregate stats + Mann-Whitney U tests across multiseed benchmark runs.

Reads results/multiseed/{optimizer}_seed{N}/results.jsonl and produces:
  - A markdown table on stdout
  - results/multiseed/summary.json for downstream consumption (README table,
    tests, future plots)

Metrics (for each optimizer, aggregated across seeds):
  - fast_cluster_count  — # trials with avg_latency_ms < --fast-threshold-ms
  - post_hit_consistency — fraction of trials after the first fast hit that
    stayed in the fast cluster (1.0 = locked in immediately, 0.0 = never
    revisits fast)
  - best_latency_ms — min avg_latency_ms across a seed's 15 trials
  - first_fast_hit_trial — trial number where the seed first hit the fast
    cluster (lower = faster discovery)
  - feasibility — trials_feasible / trials_total
  - crashes — crash count

Mann-Whitney U tests (non-parametric, small-n safe):
  - fast_cluster_count  (TBA-TPE > Random, one-sided)
  - post_hit_consistency (TBA-TPE > Random, one-sided)
  - best_latency        (two-sided — we're not predicting direction)

Usage:
    python scripts/compute_multiseed_stats.py \\
        --results-dir results/multiseed/ \\
        --fast-threshold-ms 1000
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from scipy.stats import mannwhitneyu

_SEED_DIR_RE = re.compile(r"^(?P<opt>.+)_seed(?P<seed>\d+)$")


def _load_trials(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _seed_stats(trials: list[dict[str, Any]], fast_threshold_ms: float) -> dict[str, Any]:
    """Compute per-seed metrics from one results.jsonl."""
    trials = sorted(trials, key=lambda r: r["trial"])
    feasible = [t for t in trials if t.get("status") == "feasible"]
    crashed = [t for t in trials if t.get("status") == "crash"]

    def is_fast(t: dict[str, Any]) -> bool:
        lat = t.get("avg_latency_ms")
        return lat is not None and lat < fast_threshold_ms

    fast_trials = [t for t in trials if is_fast(t)]
    first_fast = next((t["trial"] for t in trials if is_fast(t)), None)

    post_hit = None
    if first_fast is not None:
        remaining = [t for t in trials if t["trial"] > first_fast]
        if remaining:
            post_hit = sum(1 for t in remaining if is_fast(t)) / len(remaining)
        else:
            post_hit = 1.0  # no remaining trials — vacuously consistent

    latencies = [t["avg_latency_ms"] for t in feasible if "avg_latency_ms" in t]
    return {
        "n_trials": len(trials),
        "n_feasible": len(feasible),
        "n_crashes": len(crashed),
        "fast_cluster_count": len(fast_trials),
        "post_hit_consistency": post_hit,
        "first_fast_hit_trial": first_fast,
        "best_latency_ms": min(latencies) if latencies else None,
    }


def _aggregate(per_seed: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Mean/std/min/max over seeds for one metric."""
    vals = [s[key] for s in per_seed if s[key] is not None]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
        "values": vals,
    }


def compute(results_dir: Path, fast_threshold_ms: float) -> dict[str, Any]:
    """Walk *results_dir* and return the full summary dict."""
    by_opt: dict[str, dict[int, dict[str, Any]]] = {}
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
        by_opt.setdefault(opt, {})[seed] = _seed_stats(
            _load_trials(trials_path), fast_threshold_ms,
        )

    per_opt: dict[str, dict[str, Any]] = {}
    for opt, seed_map in by_opt.items():
        seeds = sorted(seed_map.keys())
        per_seed = [seed_map[s] for s in seeds]
        per_opt[opt] = {
            "seeds": seeds,
            "per_seed": per_seed,
            "total_trials": sum(s["n_trials"] for s in per_seed),
            "total_feasible": sum(s["n_feasible"] for s in per_seed),
            "total_crashes": sum(s["n_crashes"] for s in per_seed),
            "fast_cluster_count": _aggregate(per_seed, "fast_cluster_count"),
            "post_hit_consistency": _aggregate(per_seed, "post_hit_consistency"),
            "best_latency_ms": _aggregate(per_seed, "best_latency_ms"),
            "first_fast_hit_trial": _aggregate(per_seed, "first_fast_hit_trial"),
        }

    tests: dict[str, Any] = {}
    if "random" in per_opt and "tba-tpe" in per_opt:
        r = per_opt["random"]
        t = per_opt["tba-tpe"]
        u_fast, p_fast = mannwhitneyu(
            t["fast_cluster_count"]["values"],
            r["fast_cluster_count"]["values"],
            alternative="greater",
        )
        u_post, p_post = mannwhitneyu(
            t["post_hit_consistency"]["values"],
            r["post_hit_consistency"]["values"],
            alternative="greater",
        )
        u_lat, p_lat = mannwhitneyu(
            t["best_latency_ms"]["values"],
            r["best_latency_ms"]["values"],
            alternative="two-sided",
        )
        _one_sided = "tba-tpe > random"
        tests = {
            "fast_cluster_count": {
                "U": float(u_fast), "p": float(p_fast), "alternative": _one_sided,
            },
            "post_hit_consistency": {
                "U": float(u_post), "p": float(p_post), "alternative": _one_sided,
            },
            "best_latency_ms": {
                "U": float(u_lat), "p": float(p_lat), "alternative": "two-sided",
            },
        }

    return {
        "fast_threshold_ms": fast_threshold_ms,
        "per_optimizer": per_opt,
        "mann_whitney": tests,
    }


def _fmt_mean_std(agg: dict[str, Any], precision: int = 2) -> str:
    return f"{agg['mean']:.{precision}f} ± {agg['std']:.{precision}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a compact markdown summary suitable for pasting into a writeup."""
    per_opt = summary["per_optimizer"]
    tests = summary["mann_whitney"]
    opts = [o for o in ("random", "tba-tpe") if o in per_opt]
    if len(opts) < 2:
        return "Need both random and tba-tpe runs for comparison."

    lines = [
        f"# Multi-seed comparison (fast threshold = {summary['fast_threshold_ms']:.0f} ms)",
        "",
        f"| Metric | Random (n={per_opt['random']['fast_cluster_count']['n']}) | "
        f"TBA-TPE (n={per_opt['tba-tpe']['fast_cluster_count']['n']}) | "
        f"Mann-Whitney p |",
        "|---|---|---|---|",
    ]

    rows = [
        ("Fast-cluster trials / 15", "fast_cluster_count", 2),
        ("Post-hit consistency",     "post_hit_consistency", 3),
        ("Best latency (ms)",        "best_latency_ms", 2),
        ("First fast-hit trial",     "first_fast_hit_trial", 2),
    ]
    for label, key, prec in rows:
        r = per_opt["random"][key]
        t = per_opt["tba-tpe"][key]
        p = tests.get(key, {}).get("p")
        p_str = f"{p:.3f}" if p is not None else "—"
        lines.append(
            f"| {label} | {_fmt_mean_std(r, prec)} | {_fmt_mean_std(t, prec)} | {p_str} |"
        )

    r_feas = f"{per_opt['random']['total_feasible']}/{per_opt['random']['total_trials']}"
    t_feas = f"{per_opt['tba-tpe']['total_feasible']}/{per_opt['tba-tpe']['total_trials']}"
    r_crash = per_opt["random"]["total_crashes"]
    t_crash = per_opt["tba-tpe"]["total_crashes"]
    lines.append(f"| Feasibility | {r_feas} | {t_feas} | — |")
    lines.append(f"| Crashes     | {r_crash} | {t_crash} | — |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/multiseed"))
    parser.add_argument("--fast-threshold-ms", type=float, default=1000.0)
    parser.add_argument(
        "--summary-out", type=Path, default=None,
        help="Where to write summary.json (default: <results-dir>/summary.json)",
    )
    args = parser.parse_args()

    summary = compute(args.results_dir, args.fast_threshold_ms)
    out = args.summary_out or (args.results_dir / "summary.json")
    out.write_text(json.dumps(summary, indent=2, default=list))
    print(render_markdown(summary))
    print(f"\nFull summary written to {out}")


if __name__ == "__main__":
    main()
