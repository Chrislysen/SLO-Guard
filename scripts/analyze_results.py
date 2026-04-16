#!/usr/bin/env python3
"""Analyze and compare optimizer results from SLO-Guard experiments.

Generates:
  1. Convergence plot: best latency/goodput over trials (random vs tba-tpe)
  2. Crash rate comparison bar chart
  3. Config space scatter: feasible vs crash colored by optimizer
  4. Crash pattern analysis: which config knobs predict crashes
  5. Summary table

Usage:
    python scripts/analyze_results.py \
        --random-dir results/random_curl_run/ \
        --tba-tpe-dir results/tba_tpe_run/ \
        --output figures/comparison/

    # Or load all results from a root dir:
    python scripts/analyze_results.py \
        --results-dir results/ \
        --output figures/comparison/
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("analyze")

COLORS = {
    "RandomSearchOptimizer": "#999999",
    "TBATPEHybrid": "#4CAF50",
    "OptunaColdTPE": "#2196F3",
    "TBAOptimizer": "#FF9800",
}
LABELS = {
    "RandomSearchOptimizer": "Random Search",
    "TBATPEHybrid": "TBA-TPE Hybrid",
    "OptunaColdTPE": "Optuna TPE",
    "TBAOptimizer": "TBA",
}

CONFIG_KNOBS = [
    "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
    "max_model_len", "enforce_eager", "enable_chunked_prefill",
    "enable_prefix_caching",
]


def load_jsonl(path: Path) -> list[dict]:
    """Load trials from a single JSONL file."""
    trials = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials


def load_dir(directory: Path) -> dict[str, list[list[dict]]]:
    """Load all JSONL files from a directory, grouped by optimizer."""
    data: dict[str, list[list[dict]]] = defaultdict(list)
    for p in sorted(directory.glob("**/*.jsonl")):
        trials = load_jsonl(p)
        if trials:
            opt_name = trials[0].get("optimizer_name", "unknown")
            data[opt_name].append(trials)
    return data


# ── Plot 1: Convergence ─────────────────────────────────────────────────────

def plot_convergence(data: dict[str, list[list[dict]]], output: Path) -> None:
    """Best goodput over trials for each optimizer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for opt_name, runs in sorted(data.items()):
        color = COLORS.get(opt_name, "#333333")
        label = LABELS.get(opt_name, opt_name)

        # Goodput convergence
        goodput_curves = []
        latency_curves = []
        for trials in runs:
            gp_curve, lat_curve = [], []
            best_gp, best_lat = 0.0, float("inf")
            for t in trials:
                gp = t.get("goodput_tokens_per_sec") or 0
                lat = t.get("request_latency_p50")
                feasible = t.get("feasible", False)
                if feasible and gp > best_gp:
                    best_gp = gp
                if feasible and lat is not None and lat < best_lat:
                    best_lat = lat
                gp_curve.append(best_gp)
                lat_curve.append(best_lat if best_lat < float("inf") else None)
            goodput_curves.append(gp_curve)
            latency_curves.append(lat_curve)

        if not goodput_curves:
            continue

        max_len = max(len(c) for c in goodput_curves)
        x = np.arange(1, max_len + 1)

        # Goodput
        padded = [c + [c[-1]] * (max_len - len(c)) for c in goodput_curves]
        arr = np.array(padded, dtype=float)
        mean = arr.mean(axis=0)
        ax1.plot(x, mean, color=color, label=label, linewidth=2, marker="o", markersize=4)
        if len(padded) > 1:
            std = arr.std(axis=0)
            ax1.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

        # Latency (replace None with inf for plotting, then mask)
        lat_padded = []
        for c in latency_curves:
            padded_c = c + [c[-1]] * (max_len - len(c))
            lat_padded.append([v if v is not None else np.nan for v in padded_c])
        lat_arr = np.array(lat_padded, dtype=float)
        lat_mean = np.nanmean(lat_arr, axis=0)
        valid = ~np.isnan(lat_mean)
        if valid.any():
            ax2.plot(x[valid], lat_mean[valid], color=color, label=label,
                     linewidth=2, marker="o", markersize=4)

    # Mark TBA→TPE handoff if phase data is available
    for opt_name, runs in data.items():
        if opt_name != "TBATPEHybrid":
            continue
        for trials in runs:
            for i, t in enumerate(trials):
                if t.get("optimizer_phase") == "tpe-exploit":
                    # First TPE trial = handoff point
                    for ax in (ax1, ax2):
                        ax.axvline(x=i + 1, color="#4CAF50", linestyle="--",
                                   alpha=0.6, linewidth=1.5, label="TBA→TPE handoff")
                    break

    has_latency = any(line.get_label() != "_nolegend_" for line in ax2.get_lines())
    ax1.set_xlabel("Trial", fontsize=12)
    ax1.set_ylabel("Best Goodput (tokens/s)", fontsize=12)
    ax1.set_title("Goodput Convergence", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Trial", fontsize=12)
    ax2.set_ylabel("Best Latency p50 (ms)", fontsize=12)
    ax2.set_title("Latency Convergence (lower is better)", fontsize=13)
    if has_latency:
        ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output)


# ── Plot 2: Crash Rate Comparison ────────────────────────────────────────────

def plot_crash_rates(data: dict[str, list[list[dict]]], output: Path) -> None:
    """Bar chart comparing crash/feasible/infeasible rates."""
    fig, ax = plt.subplots(figsize=(10, 5))

    opt_names = sorted(data.keys())
    labels = [LABELS.get(n, n) for n in opt_names]
    colors = [COLORS.get(n, "#333") for n in opt_names]
    x = np.arange(len(opt_names))
    width = 0.25

    crash_means, feas_means, infeas_means = [], [], []
    for name in opt_names:
        crashes, feasibles, infeasibles = [], [], []
        for trials in data[name]:
            n = len(trials)
            c = sum(1 for t in trials if t.get("crashed"))
            f = sum(1 for t in trials if t.get("feasible"))
            inf = n - c - f
            crashes.append(c / n * 100 if n else 0)
            feasibles.append(f / n * 100 if n else 0)
            infeasibles.append(inf / n * 100 if n else 0)
        crash_means.append(np.mean(crashes))
        feas_means.append(np.mean(feasibles))
        infeas_means.append(np.mean(infeasibles))

    bars1 = ax.bar(x - width, feas_means, width, label="Feasible", color="#4CAF50", alpha=0.8)
    bars2 = ax.bar(x, infeas_means, width, label="Infeasible", color="#FF9800", alpha=0.8)
    bars3 = ax.bar(x + width, crash_means, width, label="Crash", color="#F44336", alpha=0.8)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        f"{h:.0f}%", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("% of Trials", fontsize=12)
    ax.set_title("Trial Outcomes by Optimizer", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output)


# ── Plot 3: Config Space Scatter ─────────────────────────────────────────────

def plot_config_scatter(data: dict[str, list[list[dict]]], output: Path) -> None:
    """Config space scatter: 2x2 projections colored by outcome."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    all_trials = []
    for opt_name, runs in data.items():
        for trials in runs:
            for t in trials:
                t["_opt"] = opt_name
                all_trials.append(t)

    projections = [
        ("gpu_memory_utilization", "max_num_seqs", False, True),
        ("max_model_len", "max_num_seqs", True, True),
        ("gpu_memory_utilization", "max_model_len", False, True),
        ("enforce_eager", "max_model_len", False, True),
    ]

    for ax, (xkey, ykey, xlog, ylog) in zip(axes.flat, projections):
        for outcome, color, marker, label in [
            ("feasible", "#4CAF50", "o", "Feasible"),
            ("infeasible", "#FF9800", "s", "Infeasible"),
            ("crashed", "#F44336", "x", "Crash"),
        ]:
            xs, ys = [], []
            for t in all_trials:
                config = t.get("config", {})
                x_val = config.get(xkey)
                y_val = config.get(ykey)
                if x_val is None or y_val is None:
                    continue

                is_match = False
                if outcome == "crashed" and t.get("crashed"):
                    is_match = True
                elif outcome == "feasible" and t.get("feasible"):
                    is_match = True
                elif outcome == "infeasible" and not t.get("crashed") and not t.get("feasible"):
                    is_match = True

                if is_match:
                    # Jitter boolean values for visibility
                    if isinstance(x_val, bool):
                        x_val = int(x_val) + np.random.uniform(-0.15, 0.15)
                    if isinstance(y_val, bool):
                        y_val = int(y_val) + np.random.uniform(-0.15, 0.15)
                    xs.append(x_val)
                    ys.append(y_val)

            if xs:
                ax.scatter(xs, ys, c=color, marker=marker, s=50, alpha=0.7, label=label)

        ax.set_xlabel(xkey, fontsize=10)
        ax.set_ylabel(ykey, fontsize=10)
        if xlog:
            ax.set_xscale("log")
        if ylog and ykey in ("max_num_seqs", "max_model_len", "max_num_batched_tokens"):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # One legend for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Config Space Exploration: Feasible vs Crash Regions", fontsize=14, y=1.05)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output)


# ── Plot 4: Crash Pattern Analysis ──────────────────────────────────────────

def analyze_crash_patterns(data: dict[str, list[list[dict]]], output: Path) -> str:
    """Analyze which config knobs predict crashes vs feasible outcomes.

    Generates a feature importance plot and prints decision rules.
    """
    feasible_configs = []
    crash_configs = []

    for runs in data.values():
        for trials in runs:
            for t in trials:
                config = t.get("config", {})
                if t.get("feasible"):
                    feasible_configs.append(config)
                elif t.get("crashed"):
                    crash_configs.append(config)

    if not feasible_configs and not crash_configs:
        return "No data to analyze."

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("  CRASH PATTERN ANALYSIS")
    report_lines.append("=" * 60)
    report_lines.append(f"  Feasible configs: {len(feasible_configs)}")
    report_lines.append(f"  Crash configs:    {len(crash_configs)}")
    report_lines.append("")

    # Compare distributions per knob
    numeric_knobs = [
        "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
        "max_model_len",
    ]
    bool_knobs = ["enforce_eager", "enable_chunked_prefill", "enable_prefix_caching"]

    n_plots = len(numeric_knobs) + len(bool_knobs)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))

    # Hide unused axes
    for idx in range(n_plots, nrows * ncols):
        axes.flat[idx].set_visible(False)

    # Numeric knobs: box plots
    for i, knob in enumerate(numeric_knobs):
        ax = axes.flat[i]
        feas_vals = [c.get(knob) for c in feasible_configs if c.get(knob) is not None]
        crash_vals = [c.get(knob) for c in crash_configs if c.get(knob) is not None]

        if feas_vals or crash_vals:
            bp_data = [feas_vals if feas_vals else [0], crash_vals if crash_vals else [0]]
            bp = ax.boxplot(bp_data, tick_labels=["Feasible", "Crash"], patch_artist=True)
            bp["boxes"][0].set_facecolor("#4CAF50")
            bp["boxes"][0].set_alpha(0.6)
            if len(bp["boxes"]) > 1:
                bp["boxes"][1].set_facecolor("#F44336")
                bp["boxes"][1].set_alpha(0.6)

            # Stats
            if feas_vals and crash_vals:
                f_mean, c_mean = np.mean(feas_vals), np.mean(crash_vals)
                report_lines.append(
                    f"  {knob}: feasible mean={f_mean:.1f}, crash mean={c_mean:.1f}"
                    f" (diff={abs(c_mean - f_mean):.1f})"
                )

        ax.set_title(knob, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    # Boolean knobs: crash rate by value
    for i, knob in enumerate(bool_knobs):
        ax = axes.flat[i + len(numeric_knobs)]
        if i + len(numeric_knobs) >= len(axes.flat):
            break

        true_feas = sum(1 for c in feasible_configs if c.get(knob) is True)
        true_crash = sum(1 for c in crash_configs if c.get(knob) is True)
        false_feas = sum(1 for c in feasible_configs if c.get(knob) is False)
        false_crash = sum(1 for c in crash_configs if c.get(knob) is False)
        # Count configs where knob is absent (conditional variable)
        absent_feas = sum(1 for c in feasible_configs if knob not in c)
        absent_crash = sum(1 for c in crash_configs if knob not in c)

        values = ["True", "False"]
        feas_counts = [true_feas, false_feas]
        crash_counts = [true_crash, false_crash]

        if absent_feas + absent_crash > 0:
            values.append("N/A")
            feas_counts.append(absent_feas)
            crash_counts.append(absent_crash)

        x_pos = np.arange(len(values))
        ax.bar(x_pos - 0.15, feas_counts, 0.3, label="Feasible", color="#4CAF50", alpha=0.7)
        ax.bar(x_pos + 0.15, crash_counts, 0.3, label="Crash", color="#F44336", alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(values)
        ax.set_title(knob, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        total_true = true_feas + true_crash
        total_false = false_feas + false_crash
        if total_true > 0:
            crash_rate_true = true_crash / total_true * 100
            report_lines.append(
                f"  {knob}=True: crash rate {crash_rate_true:.0f}% ({true_crash}/{total_true})"
            )
        if total_false > 0:
            crash_rate_false = false_crash / total_false * 100
            report_lines.append(
                f"  {knob}=False: crash rate {crash_rate_false:.0f}% ({false_crash}/{total_false})"
            )

    fig.suptitle("Crash Patterns: Feasible vs Crash Config Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output)

    # Interaction analysis: enforce_eager x max_model_len
    report_lines.append("")
    report_lines.append("  INTERACTION: enforce_eager x max_model_len")
    report_lines.append("  " + "-" * 50)

    combos = defaultdict(lambda: {"feasible": 0, "crash": 0})
    for c in feasible_configs:
        eager = c.get("enforce_eager", "N/A")
        mlen = c.get("max_model_len", 0)
        bucket = "<=1024" if mlen <= 1024 else ">1024"
        combos[(eager, bucket)]["feasible"] += 1
    for c in crash_configs:
        eager = c.get("enforce_eager", "N/A")
        mlen = c.get("max_model_len", 0)
        bucket = "<=1024" if mlen <= 1024 else ">1024"
        combos[(eager, bucket)]["crash"] += 1

    for (eager, bucket), counts in sorted(combos.items()):
        total = counts["feasible"] + counts["crash"]
        crash_pct = counts["crash"] / total * 100 if total else 0
        report_lines.append(
            f"    enforce_eager={eager}, max_model_len {bucket}: "
            f"{counts['crash']}/{total} crash ({crash_pct:.0f}%)"
        )

    # Interaction: gpu_mem x max_num_seqs
    report_lines.append("")
    report_lines.append("  INTERACTION: gpu_memory_utilization x max_num_seqs")
    report_lines.append("  " + "-" * 50)

    combos2 = defaultdict(lambda: {"feasible": 0, "crash": 0})
    for c in feasible_configs:
        gpu = c.get("gpu_memory_utilization", 0)
        seqs = c.get("max_num_seqs", 0)
        gpu_bucket = "<=0.75" if gpu <= 0.75 else ">0.75"
        seq_bucket = "<=32" if seqs <= 32 else ">32"
        combos2[(gpu_bucket, seq_bucket)]["feasible"] += 1
    for c in crash_configs:
        gpu = c.get("gpu_memory_utilization", 0)
        seqs = c.get("max_num_seqs", 0)
        gpu_bucket = "<=0.75" if gpu <= 0.75 else ">0.75"
        seq_bucket = "<=32" if seqs <= 32 else ">32"
        combos2[(gpu_bucket, seq_bucket)]["crash"] += 1

    for (gpu, seqs), counts in sorted(combos2.items()):
        total = counts["feasible"] + counts["crash"]
        crash_pct = counts["crash"] / total * 100 if total else 0
        report_lines.append(
            f"    gpu_mem {gpu}, max_num_seqs {seqs}: "
            f"{counts['crash']}/{total} crash ({crash_pct:.0f}%)"
        )

    report_lines.append("")
    report_lines.append("=" * 60)

    report = "\n".join(report_lines)
    print(report)
    return report


# ── Summary Table ────────────────────────────────────────────────────────────

def print_summary(data: dict[str, list[list[dict]]]) -> str:
    header = (
        f"{'Method':<25} {'Best Goodput':>12} {'Best Lat p50':>12} "
        f"{'Crash%':>8} {'Feasible%':>10} {'1st Feas':>8}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for opt_name in sorted(data.keys()):
        label = LABELS.get(opt_name, opt_name)
        runs = data[opt_name]

        best_gps, best_lats, crash_pcts, feas_pcts, first_feas = [], [], [], [], []
        for trials in runs:
            n = len(trials)
            crashes = sum(1 for t in trials if t.get("crashed"))
            feasible = sum(1 for t in trials if t.get("feasible"))
            crash_pcts.append(crashes / n * 100 if n else 0)
            feas_pcts.append(feasible / n * 100 if n else 0)

            best_gp, best_lat, ff = 0.0, float("inf"), n + 1
            for i, t in enumerate(trials):
                gp = t.get("goodput_tokens_per_sec") or 0
                lat = t.get("request_latency_p50")
                if t.get("feasible"):
                    if gp > best_gp:
                        best_gp = gp
                    if lat is not None and lat < best_lat:
                        best_lat = lat
                    if ff == n + 1:
                        ff = i + 1
            best_gps.append(best_gp)
            best_lats.append(best_lat if best_lat < float("inf") else 0)
            first_feas.append(ff)

        lat_str = f"{np.mean(best_lats):.0f}ms" if any(l > 0 for l in best_lats) else "N/A"
        ff_str = f"{np.mean(first_feas):.0f}" if any(f <= max(len(r) for r in runs) for f in first_feas) else "N/A"

        lines.append(
            f"{label:<25} "
            f"{np.mean(best_gps):>10.1f}  "
            f"{lat_str:>12} "
            f"{np.mean(crash_pcts):>6.1f}%  "
            f"{np.mean(feas_pcts):>8.1f}%  "
            f"{ff_str:>8}"
        )

    lines.append(sep)
    table = "\n".join(lines)
    print(table)
    return table


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze and compare SLO-Guard results")
    parser.add_argument("--results-dir", help="Root results directory (loads all JSONL)")
    parser.add_argument("--random-dir", help="Random search results directory")
    parser.add_argument("--tba-tpe-dir", help="TBA-TPE results directory")
    parser.add_argument("--output", default="figures/comparison/", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data: dict[str, list[list[dict]]] = defaultdict(list)

    if args.results_dir:
        data.update(load_dir(Path(args.results_dir)))
    if args.random_dir:
        for loaded_data in load_dir(Path(args.random_dir)).values():
            data["RandomSearchOptimizer"].extend(loaded_data)
    if args.tba_tpe_dir:
        for loaded_data in load_dir(Path(args.tba_tpe_dir)).values():
            data["TBATPEHybrid"].extend(loaded_data)

    if not data:
        print("No experiment data found. Provide --results-dir or --random-dir/--tba-tpe-dir.")
        return

    total_trials = sum(len(t) for runs in data.values() for t in runs)
    logger.info(
        "Loaded %d runs across %d optimizers (%d total trials)",
        sum(len(v) for v in data.values()), len(data), total_trials,
    )

    # Generate all outputs
    print()
    print_summary(data)
    print()

    plot_convergence(data, output_dir / "convergence.png")
    plot_crash_rates(data, output_dir / "crash_rates.png")
    plot_config_scatter(data, output_dir / "config_scatter.png")

    report = analyze_crash_patterns(data, output_dir / "crash_patterns.png")

    # Save report to text file
    report_path = output_dir / "crash_analysis.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Saved crash analysis: %s", report_path)

    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
