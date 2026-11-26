#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh
# Regenerates EVERY table and figure in the paper.
# Run time: ~20–40 min on a modern CPU, less on GPU.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-config/default.yaml}"
DATA_DIR="${DATA_DIR:-data/processed}"
CKPT="artifacts/checkpoints/nqal_final.pt"
RESULTS="artifacts/results"

mkdir -p "$RESULTS" artifacts/logs artifacts/checkpoints

# 0. Activate env
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# 1. Dataset
if [[ ! -d "$DATA_DIR" ]] || [[ ! -f "$DATA_DIR/X_train.npy" ]]; then
  echo "==> Generating dataset..."
  python data/generate_dataset.py --config "$CONFIG" --seed 42 --out-dir "$DATA_DIR"
fi

# 2. Training
if [[ ! -f "$CKPT" ]]; then
  echo "==> Training NQAL..."
  python train.py --config "$CONFIG" --data-dir "$DATA_DIR"
fi

# 3. Main evaluation
echo "==> Evaluating (accuracy, FPR, attacks, hardware)"
python evaluate.py --config "$CONFIG" --data-dir "$DATA_DIR" --checkpoint "$CKPT" --output-dir "$RESULTS"

# 4. Tables
echo "==> Table 1"
python experiments/table1_comparison.py --out "$RESULTS/tables"

echo "==> Table 2"
python experiments/table2_taxonomy.py --config "$CONFIG" --data-dir "$DATA_DIR" --checkpoint "$CKPT" --out "$RESULTS/tables"

# 5. Figure 1
echo "==> Figure 1 (architecture)"
python experiments/figure1_architecture.py --out "$RESULTS/figures"

# 6. Robustness
echo "==> Robustness sweep"
python experiments/robustness_study.py --config "$CONFIG" --data-dir "$DATA_DIR" --checkpoint "$CKPT" --out "$RESULTS"

# 7. Federated efficiency
echo "==> Federated efficiency"
python experiments/federated_efficiency.py --config "$CONFIG" --data-dir "$DATA_DIR" --out "$RESULTS"

# 8. Real-time benchmark
echo "==> Real-time benchmark"
python experiments/real_time_benchmark.py --config "$CONFIG" --data-dir "$DATA_DIR" --checkpoint "$CKPT" --out "$RESULTS"

# 9. ZKP privacy audit
echo "==> ZKP privacy audit"
python experiments/zkp_privacy_audit.py --config "$CONFIG" --checkpoint "$CKPT" --out "$RESULTS"

# 10. Theorems
echo "==> Theorems 1 & 2"
python experiments/theorems.py --config "$CONFIG" --checkpoint "$CKPT" --out "$RESULTS"

echo
echo "=============================================================="
echo " Reproduction complete. Artifacts:"
echo "   Tables:  $RESULTS/tables/"
echo "   Figures: $RESULTS/figures/  +  $RESULTS/*.png"
echo "   JSON:    $RESULTS/*.json"
echo "=============================================================="
