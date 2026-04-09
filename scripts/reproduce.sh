#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "  SLO-Guard Full Reproduction Script"
echo "=========================================="
echo ""
echo "Estimated runtime: ~48 GPU-hours"
echo "Requires: vLLM, GPU with >=16GB VRAM"
echo ""

# 1. Install
echo "[1/5] Installing SLO-Guard..."
pip install -e ".[all]" --quiet

# 2. Download models
echo "[2/5] Downloading models..."
python scripts/download_models.py

# 3. Run experiments
echo "[3/5] Running experiments..."

MODELS=("Qwen/Qwen2-1.5B" "microsoft/phi-2" "mistralai/Mistral-7B-v0.1")
WORKLOADS=("interactive" "batch" "bursty")
METHODS=("random" "tpe" "tba" "tba-tpe")
SEEDS="0,1,2,3,4"
BUDGET=30

for workload in "${WORKLOADS[@]}"; do
  for model in "${MODELS[@]}"; do
    model_short=$(echo "$model" | sed 's/.*\///')
    for method in "${METHODS[@]}"; do
      echo ""
      echo "  Running: $method | $workload | $model_short"
      python scripts/run_experiment.py \
        --model "$model" \
        --optimizer "$method" \
        --workload "$workload" \
        --budget "$BUDGET" \
        --seeds "$SEEDS" \
        --output "results/${workload}/${model_short}/${method}" \
        2>&1 | tail -3
    done
  done
done

# 4. Generate figures
echo ""
echo "[4/5] Generating figures..."
sloguard report --results-dir results/ --output figures/

# 5. Summary
echo ""
echo "[5/5] Done!"
echo ""
echo "Results:  results/"
echo "Figures:  figures/"
echo ""
echo "Key files:"
echo "  figures/goodput_convergence.png"
echo "  figures/latency_dist.png"
echo "  figures/crash_scatter.png"
echo "  figures/crash_waste.png"
