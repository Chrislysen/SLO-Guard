"""CLI entry point for SLO-Guard.

Commands:
  sloguard tune    — Run a benchmark/tuning experiment
  sloguard report  — Generate plots from experiment logs
  sloguard list-optimizers — Show available optimizers
"""
from __future__ import annotations

import logging
import sys

import click

from sloguard import __version__


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool) -> None:
    """SLO-Guard: Crash-aware autotuning for LLM serving."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.option("--model", required=True, help="HuggingFace model ID (e.g. Qwen/Qwen2-1.5B)")
@click.option("--optimizer", default="tba-tpe",
              type=click.Choice(["random", "tpe", "tba", "tba-tpe", "constrained-bo"]),
              help="Optimizer to use")
@click.option("--budget", default=30, type=int, help="Number of trials")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--slo-ttft-p99", default=500.0, type=float, help="TTFT p99 SLO (ms)")
@click.option("--slo-itl-p99", default=100.0, type=float, help="ITL p99 SLO (ms)")
@click.option("--slo-latency-p99", default=30000.0, type=float, help="Request latency p99 SLO (ms)")
@click.option("--slo-memory-mb", default=0.0, type=float, help="GPU memory SLO (MB, 0=no limit)")
@click.option("--workload", default="interactive",
              type=click.Choice(["interactive", "batch", "bursty"]),
              help="Workload type")
@click.option("--request-rate", default=4.0, type=float, help="Requests per second")
@click.option("--num-requests", default=100, type=int, help="Total requests per trial")
@click.option("--output", default="results", help="Output directory")
@click.option("--port", default=8000, type=int, help="vLLM server port")
@click.option("--gpu-id", default="auto", help="GPU identifier for logging")
def tune(
    model: str,
    optimizer: str,
    budget: int,
    seed: int,
    slo_ttft_p99: float,
    slo_itl_p99: float,
    slo_latency_p99: float,
    slo_memory_mb: float,
    workload: str,
    request_rate: float,
    num_requests: int,
    output: str,
    port: int,
    gpu_id: str,
) -> None:
    """Run a serving configuration tuning experiment."""
    from sloguard.config_space import build_serving_space
    from sloguard.experiment_runner import ExperimentRunner, create_optimizer
    from sloguard.load_generator import WorkloadConfig
    from sloguard.slo_contract import SLOContract

    # Build SLO contract
    slo = SLOContract(
        ttft_p99_ms=slo_ttft_p99,
        itl_p99_ms=slo_itl_p99,
        request_latency_p99_ms=slo_latency_p99,
        gpu_memory_mb=slo_memory_mb,
    )

    # Build search space
    space = build_serving_space()

    # Build optimizer
    constraints = slo.to_constraints_dict()
    opt = create_optimizer(optimizer, space, constraints, budget, seed)

    # Workload config
    workload_presets = {
        "interactive": {
            "request_rate": request_rate,
            "num_requests": num_requests,
            "prompt_len_min": 128,
            "prompt_len_max": 512,
            "output_len_min": 64,
            "output_len_max": 256,
        },
        "batch": {
            "request_rate": request_rate * 4,
            "num_requests": num_requests,
            "prompt_len_min": 512,
            "prompt_len_max": 2048,
            "output_len_min": 256,
            "output_len_max": 1024,
        },
        "bursty": {
            "request_rate": request_rate,
            "num_requests": num_requests,
            "prompt_len_min": 128,
            "prompt_len_max": 512,
            "output_len_min": 64,
            "output_len_max": 256,
        },
    }

    preset = workload_presets[workload]
    wl_config = WorkloadConfig(
        request_rate=preset["request_rate"],
        num_requests=preset["num_requests"],
        prompt_len_min=preset["prompt_len_min"],
        prompt_len_max=preset["prompt_len_max"],
        output_len_min=preset["output_len_min"],
        output_len_max=preset["output_len_max"],
        model=model,
    )

    workload_mode = "burst" if workload == "bursty" else "fixed"
    workload_kwargs = {}
    if workload == "bursty":
        workload_kwargs = {
            "baseline_rate": request_rate,
            "peak_rate": request_rate * 5,
        }

    # Run experiment
    click.echo(f"SLO-Guard v{__version__}")
    click.echo(f"Model:     {model}")
    click.echo(f"Optimizer: {optimizer}")
    click.echo(f"Budget:    {budget} trials")
    click.echo(f"SLOs:      TTFT_p99<={slo_ttft_p99}ms, ITL_p99<={slo_itl_p99}ms")
    click.echo(f"Workload:  {workload} @ {preset['request_rate']} req/s")
    click.echo()

    runner = ExperimentRunner(
        model=model,
        optimizer=opt,
        slo=slo,
        workload=wl_config,
        output_dir=output,
        gpu_id=gpu_id,
        port=port,
        workload_mode=workload_mode,
        workload_kwargs=workload_kwargs,
    )

    best = runner.run(budget)

    if best is not None:
        config, result = best
        click.echo()
        click.echo("=" * 60)
        click.echo("BEST FEASIBLE CONFIG:")
        click.echo("=" * 60)
        for k, v in config.items():
            click.echo(f"  {k}: {v}")
        click.echo()
        click.echo(f"  Goodput: {result.goodput_tokens_per_sec or 0:.1f} tok/s")
        click.echo(f"  TTFT p99: {result.ttft_p99_ms or 0:.1f} ms")
        click.echo(f"  ITL p99: {result.itl_p99_ms or 0:.1f} ms")
        click.echo(f"  Crash waste: {runner.optimizer.n_crashes}/{budget} trials")
    else:
        click.echo()
        click.echo("No feasible configuration found within budget.")
        sys.exit(1)


@main.command()
@click.option("--results-dir", default="results", help="Directory with JSONL experiment logs")
@click.option("--output", default="figures", help="Output directory for plots")
def report(results_dir: str, output: str) -> None:
    """Generate plots and summary from experiment logs."""
    from sloguard.report_generator import ReportGenerator

    gen = ReportGenerator(results_dir)
    gen.load_all()

    if not gen.data:
        click.echo("No experiment data found. Run 'sloguard tune' first.")
        return

    gen.generate_all(output)
    click.echo(f"Reports saved to {output}/")


@main.command("list-optimizers")
def list_optimizers() -> None:
    """List available optimizers."""
    optimizers = [
        ("random", "Uniform random sampling (baseline)"),
        ("tpe", "Optuna TPE, cold start, no crash awareness (baseline)"),
        ("tba", "TBA — crash-aware SA with feasible-region TPE"),
        ("tba-tpe", "TBA-TPE Hybrid — SA exploration + Optuna TPE exploitation"),
        ("constrained-bo", "Constrained BO with GP surrogates (requires botorch)"),
    ]
    click.echo("Available optimizers:")
    click.echo()
    for name, desc in optimizers:
        click.echo(f"  {name:<18} {desc}")


if __name__ == "__main__":
    main()
