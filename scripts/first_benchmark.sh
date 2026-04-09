#!/bin/bash
# First real SLO-Guard benchmark: 15 trials, random search, Qwen2-1.5B
set -e
source ~/sloguard-env/bin/activate
cd /mnt/c/Users/chris/VSCODE/SLO-GUARD

echo "============================================"
echo "  SLO-Guard First Benchmark"
echo "  Model: Qwen/Qwen2-1.5B"
echo "  Optimizer: random"
echo "  Budget: 15 trials"
echo "  Workload: interactive @ 2 req/s"
echo "  Requests per trial: 10"
echo "============================================"
echo ""

rm -rf results/first_run/

# Use PYTHONUNBUFFERED for real-time output
export PYTHONUNBUFFERED=1

sloguard --verbose tune \
  --model Qwen/Qwen2-1.5B \
  --optimizer random \
  --budget 15 \
  --seed 42 \
  --workload interactive \
  --request-rate 2.0 \
  --num-requests 10 \
  --slo-ttft-p99 2000 \
  --slo-itl-p99 200 \
  --slo-latency-p99 30000 \
  --output results/first_run/ \
  --port 8000

echo ""
echo "============================================"
echo "  Generating reports..."
echo "============================================"

sloguard report --results-dir results/first_run/ --output figures/first_run/

echo ""
echo "Done! Check results/first_run/ and figures/first_run/"
