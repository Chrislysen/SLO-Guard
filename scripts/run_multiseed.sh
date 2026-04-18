#!/usr/bin/env bash
# Canonical invocation for the multi-seed benchmark on Colab / Vertex AI.
#
# The orchestrator is resumable: if the VM disconnects, re-running this
# command picks up where the previous run left off (per-pair JSONL state is
# scanned and replayed into each optimizer before new trials are issued).
#
# Typical Colab cell:
#
#   %%bash
#   cd /content/slo-guard
#   pip install -e .
#   bash scripts/run_multiseed.sh
#
# To override defaults, either edit this file or call the Python script
# directly:
#
#   python scripts/run_multiseed.py \
#       --output-dir results/multiseed_concurrent/ \
#       --seeds 42 142 \
#       --optimizers tba-tpe \
#       --budget 5

set -euo pipefail

# Repo-relative so this works from the project root on Colab.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-results/multiseed_concurrent/}"
MODEL="${MODEL:-Qwen/Qwen2-1.5B}"
BUDGET="${BUDGET:-15}"

python scripts/run_multiseed.py \
    --output-dir "$OUTPUT_DIR" \
    --seeds 42 142 242 342 442 \
    --optimizers random tba-tpe \
    --budget "$BUDGET" \
    --model "$MODEL"
