"""Report generator — produces paper-quality plots from experiment logs.

Reads JSONL trial logs and generates matplotlib figures:
  1. Goodput convergence curves (all methods, with CI bands)
  2. Latency distributions (violin plots)
  3. Crash/infeasible/feasible scatter in config space
  4. Summary table
  5. Feasibility boundary heatmap
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from sloguard.trial_logger import TrialLogger

logger = logging.getLogger(__name__)

# Consistent colors per optimizer
COLORS = {
    "RandomSearchOptimizer": "#999999",
    "OptunaColdTPE": "#2196F3",
    "TBAOptimizer": "#FF9800",
    "TBATPEHybrid": "#4CAF50",
    "ConstrainedBOOptimizer": "#9C27B0",
    "random": "#999999",
    "tpe": "#2196F3",
    "tba": "#FF9800",
    "tba-tpe": "#4CAF50",
    "constrained-bo": "#9C27B0",
}

LABELS = {
    "RandomSearchOptimizer": "Random Search",
    "OptunaColdTPE": "Optuna TPE (cold)",
    "TBAOptimizer": "TBA (ours)",
    "TBATPEHybrid": "TBA-TPE Hybrid (ours)",
    "ConstrainedBOOptimizer": "Constrained BO",
    "random": "Random Search",
    "tpe": "Optuna TPE (cold)",
    "tba": "TBA (ours)",
    "tba-tpe": "TBA-TPE Hybrid (ours)",
    "constrained-bo": "Constrained BO",
}


class ReportGenerator:
    """Generates paper-quality plots from experiment JSONL logs.

    Usage:
        gen = ReportGenerator(results_dir="results/")
        gen.load_all()
        gen.plot_goodput_convergence("figures/goodput_convergence.png")
        gen.plot_latency_distributions("figures/latency_dist.png")
        gen.plot_crash_scatter("figures/crash_scatter.png")
        gen.print_summary_table()
    """

    def __init__(self, results_dir: str | Path = "results"):
        self.results_dir = Path(results_dir)
        # {optimizer_name: [list of experiment trial lists]}
        self.data: dict[str, list[list[dict]]] = defaultdict(list)

    def load_all(self) -> None:
        """Load all JSONL files from results directory."""
        if not self.results_dir.exists():
            logger.warning("Results directory not found: %s", self.results_dir)
            return

        for jsonl_path in sorted(self.results_dir.glob("**/*.jsonl")):
            trials = TrialLogger(jsonl_path).load_dicts()
            if not trials:
                continue

            optimizer_name = trials[0].get("optimizer_name", "unknown")
            self.data[optimizer_name].append(trials)

        logger.info(
            "Loaded %d experiment runs across %d optimizers",
            sum(len(v) for v in self.data.values()),
            len(self.data),
        )

    def plot_goodput_convergence(self, output_path: str = "figures/goodput_convergence.png") -> None:
        """Plot goodput vs trial number for all methods."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))

        for opt_name, runs in sorted(self.data.items()):
            color = COLORS.get(opt_name, "#333333")
            label = LABELS.get(opt_name, opt_name)

            # For each run, compute cumulative best goodput
            all_curves = []
            for trials in runs:
                curve = []
                best_so_far = 0.0
                for t in trials:
                    gp = t.get("goodput_tokens_per_sec")
                    feasible = t.get("feasible", False)
                    if feasible and gp is not None and gp > best_so_far:
                        best_so_far = gp
                    curve.append(best_so_far)
                all_curves.append(curve)

            if not all_curves:
                continue

            # Pad to same length
            max_len = max(len(c) for c in all_curves)
            padded = []
            for c in all_curves:
                padded.append(c + [c[-1]] * (max_len - len(c)))

            arr = np.array(padded)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            x = np.arange(1, max_len + 1)

            ax.plot(x, mean, color=color, label=label, linewidth=2)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_xlabel("Trial", fontsize=12)
        ax.set_ylabel("Best Goodput (tokens/s)", fontsize=12)
        ax.set_title("Goodput Convergence", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", output_path)

    def plot_latency_distributions(self, output_path: str = "figures/latency_dist.png") -> None:
        """Plot TTFT and ITL distributions as violin plots."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        opt_names = sorted(self.data.keys())
        labels = [LABELS.get(n, n) for n in opt_names]
        colors = [COLORS.get(n, "#333333") for n in opt_names]

        # TTFT p99 values
        ttft_data = []
        for name in opt_names:
            vals = []
            for runs in self.data[name]:
                for t in runs:
                    if t.get("feasible") and t.get("ttft_p99") is not None:
                        vals.append(t["ttft_p99"])
            ttft_data.append(vals if vals else [0])

        # ITL p99 values
        itl_data = []
        for name in opt_names:
            vals = []
            for runs in self.data[name]:
                for t in runs:
                    if t.get("feasible") and t.get("itl_p99") is not None:
                        vals.append(t["itl_p99"])
            itl_data.append(vals if vals else [0])

        # TTFT violin
        ax = axes[0]
        parts = ax.violinplot(ttft_data, showmedians=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i] if i < len(colors) else "#333")
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("TTFT p99 (ms)", fontsize=11)
        ax.set_title("Time to First Token", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        # ITL violin
        ax = axes[1]
        parts = ax.violinplot(itl_data, showmedians=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i] if i < len(colors) else "#333")
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("ITL p99 (ms)", fontsize=11)
        ax.set_title("Inter-Token Latency", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", output_path)

    def plot_crash_scatter(self, output_path: str = "figures/crash_scatter.png") -> None:
        """Plot feasible/infeasible/crash outcomes in config space projection."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 7))

        # Collect all trials across all optimizers
        feasible_x, feasible_y = [], []
        infeasible_x, infeasible_y = [], []
        crash_x, crash_y = [], []

        for opt_name, runs in self.data.items():
            for trials in runs:
                for t in trials:
                    config = t.get("config", {})
                    x = config.get("gpu_memory_utilization", 0.7)
                    y = config.get("max_num_seqs", 32)

                    if t.get("crashed"):
                        crash_x.append(x)
                        crash_y.append(y)
                    elif t.get("feasible"):
                        feasible_x.append(x)
                        feasible_y.append(y)
                    else:
                        infeasible_x.append(x)
                        infeasible_y.append(y)

        ax.scatter(feasible_x, feasible_y, c="#4CAF50", marker="o", s=40,
                   alpha=0.6, label=f"Feasible ({len(feasible_x)})", zorder=3)
        ax.scatter(infeasible_x, infeasible_y, c="#FF9800", marker="s", s=40,
                   alpha=0.6, label=f"Infeasible ({len(infeasible_x)})", zorder=2)
        ax.scatter(crash_x, crash_y, c="#F44336", marker="x", s=60,
                   alpha=0.8, label=f"Crash ({len(crash_x)})", zorder=4)

        ax.set_xlabel("GPU Memory Utilization", fontsize=12)
        ax.set_ylabel("Max Num Sequences", fontsize=12)
        ax.set_title("Config Space Exploration", fontsize=14)
        ax.legend(fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", output_path)

    def plot_crash_waste(self, output_path: str = "figures/crash_waste.png") -> None:
        """Bar chart of crash waste % per optimizer."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))

        opt_names = sorted(self.data.keys())
        labels = [LABELS.get(n, n) for n in opt_names]
        colors = [COLORS.get(n, "#333333") for n in opt_names]

        means, stds = [], []
        for name in opt_names:
            waste_rates = []
            for trials in self.data[name]:
                total = len(trials)
                crashes = sum(1 for t in trials if t.get("crashed"))
                waste_rates.append(crashes / total * 100 if total > 0 else 0)
            means.append(np.mean(waste_rates) if waste_rates else 0)
            stds.append(np.std(waste_rates) if waste_rates else 0)

        x = np.arange(len(opt_names))
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Crash Waste (%)", fontsize=11)
        ax.set_title("Budget Wasted on Crashes", fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{m:.1f}%", ha="center", fontsize=9)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", output_path)

    def print_summary_table(self) -> str:
        """Print and return a summary table of all methods."""
        header = (
            f"{'Method':<25} {'Best Goodput':>12} {'Crash%':>8} "
            f"{'Feasible%':>10} {'1st Feasible':>12}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for opt_name in sorted(self.data.keys()):
            runs = self.data[opt_name]
            label = LABELS.get(opt_name, opt_name)

            best_goodputs = []
            crash_rates = []
            feasible_rates = []
            first_feasibles = []

            for trials in runs:
                total = len(trials)
                crashes = sum(1 for t in trials if t.get("crashed"))
                feasible = sum(1 for t in trials if t.get("feasible"))
                crash_rates.append(crashes / total * 100 if total > 0 else 0)
                feasible_rates.append(feasible / total * 100 if total > 0 else 0)

                best_gp = 0.0
                first_feas = total  # default: never found
                for i, t in enumerate(trials):
                    gp = t.get("goodput_tokens_per_sec", 0) or 0
                    if t.get("feasible") and gp > best_gp:
                        best_gp = gp
                    if t.get("feasible") and first_feas == total:
                        first_feas = i + 1
                best_goodputs.append(best_gp)
                first_feasibles.append(first_feas)

            lines.append(
                f"{label:<25} "
                f"{np.mean(best_goodputs):>10.1f}  "
                f"{np.mean(crash_rates):>6.1f}%  "
                f"{np.mean(feasible_rates):>8.1f}%  "
                f"{np.mean(first_feasibles):>10.1f}"
            )

        lines.append(sep)
        table = "\n".join(lines)
        print(table)
        return table

    def generate_all(self, output_dir: str = "figures") -> None:
        """Generate all standard plots."""
        self.plot_goodput_convergence(f"{output_dir}/goodput_convergence.png")
        self.plot_latency_distributions(f"{output_dir}/latency_dist.png")
        self.plot_crash_scatter(f"{output_dir}/crash_scatter.png")
        self.plot_crash_waste(f"{output_dir}/crash_waste.png")
        self.print_summary_table()
