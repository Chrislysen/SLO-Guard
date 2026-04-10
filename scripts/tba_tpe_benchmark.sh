#!/bin/bash
# TBA-TPE benchmark: 15 trials on Colab A100, curl-based load generation
# Run this after first_benchmark.sh (random search) to compare optimizers
set -e
source ~/sloguard-env/bin/activate
cd /mnt/c/Users/chris/VSCODE/SLO-GUARD

echo "============================================"
echo "  SLO-Guard TBA-TPE Benchmark"
echo "  Model: Qwen/Qwen2-1.5B"
echo "  Optimizer: tba-tpe"
echo "  Budget: 15 trials"
echo "  Workload: interactive @ 2 req/s"
echo "  Requests per trial: 10"
echo "  Load gen: curl (non-streaming)"
echo "============================================"
echo ""

rm -rf results/tba_tpe_run/

export PYTHONUNBUFFERED=1

python scripts/colab_curl_benchmark.py \
  --model Qwen/Qwen2-1.5B \
  --optimizer tba-tpe \
  --budget 15 \
  --seed 42 \
  --num-requests 10 \
  --request-rate 2.0 \
  --slo-ttft-p99 2000 \
  --slo-itl-p99 200 \
  --slo-latency-p99 30000 \
  --output results/tba_tpe_run/ \
  --port 8000 \
  --verbose

echo ""
echo "============================================"
echo "  Re-running random search with curl benchmark"
echo "  (for fair comparison — same load gen method)"
echo "============================================"
echo ""

rm -rf results/random_curl_run/

python scripts/colab_curl_benchmark.py \
  --model Qwen/Qwen2-1.5B \
  --optimizer random \
  --budget 15 \
  --seed 42 \
  --num-requests 10 \
  --request-rate 2.0 \
  --slo-ttft-p99 2000 \
  --slo-itl-p99 200 \
  --slo-latency-p99 30000 \
  --output results/random_curl_run/ \
  --port 8000 \
  --verbose

echo ""
echo "============================================"
echo "  Generating comparison reports..."
echo "============================================"
echo ""

python scripts/analyze_results.py \
  --random-dir results/random_curl_run/ \
  --tba-tpe-dir results/tba_tpe_run/ \
  --output figures/comparison/

echo ""
echo "Done! Check:"
echo "  results/tba_tpe_run/    — TBA-TPE trial logs"
echo "  results/random_curl_run/ — Random trial logs"
echo "  figures/comparison/      — Comparison plots"
