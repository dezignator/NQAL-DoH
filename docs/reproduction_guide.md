# Reproduction Guide

This guide walks you through reproducing **every table, figure, and
metric** in:

> Ali, B., & Chen, G. (2025). *Neuromorphic Quantum Adversarial Learning
> (NQAL): a bio-inspired paradigm for DNS over HTTPS threat detection.*
> EURASIP Journal on Information Security, 2025:35.
> DOI: 10.1186/s13635-025-00215-5

## Prerequisites

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8 cores | 16 cores |
| RAM | 16 GB | 32 GB |
| GPU | — | CUDA ≥ 12.1, 8 GB VRAM |
| Disk | 5 GB | 20 GB |
| Python | 3.10 | 3.11 |

## One-shot reproduction

```bash
git clone https://github.com/your-org/NQAL-DoH.git
cd NQAL-DoH
bash setup_environment.sh     # venv + deps + dataset + smoke test
bash run_all_experiments.sh   # full reproduction (~30 min CPU)
```

After completion, the following files are produced:

| Artifact | What it shows |
|---|---|
| `artifacts/results/evaluation_report.json` | Accuracy, FPR, all attack accuracies, hardware metrics |
| `artifacts/results/tables/table1.md` | Comparative analysis (Table 1) |
| `artifacts/results/tables/table2.md` | Attack taxonomy + mitigation (Table 2) |
| `artifacts/results/figures/figure1_architecture.svg` | Architecture (Figure 1) |
| `artifacts/results/robustness_sweep.csv` | ε-sweep FGSM/PGD/QNI accuracy |
| `artifacts/results/robustness_curve.png` | Robustness curve |
| `artifacts/results/confusion_matrix.png` | Per-class confusion matrix |
| `artifacts/results/energy_comparison.png` | GPU vs Loihi vs Akida energy bars |
| `artifacts/results/federated_efficiency.{json,csv}` | FL comms vs accuracy trade-off |
| `artifacts/results/real_time_benchmark.json` | Latency, energy, GPU ratio |
| `artifacts/results/zkp_privacy_audit.json` | Proof size, latency, completeness, soundness |
| `artifacts/results/theorems.json` | Eq. (10), Eq. (11) empirical validation |

## Step-by-step reproduction

### Step 1 — Environment

```bash
bash setup_environment.sh
```

This creates `.venv/`, installs `requirements.txt`, generates the dataset,
and runs a 1-epoch smoke test.

### Step 2 — Dataset

```bash
python data/generate_dataset.py \
    --config config/default.yaml \
    --seed 42 \
    --out-dir data/processed
```

Output files:

```
data/processed/
├── X_train.npy  (35,000 × 50 × 4)
├── y_train.npy  (35,000,)
├── X_val.npy    (7,500 × 50 × 4)
├── y_val.npy    (7,500,)
├── X_test.npy   (7,500 × 50 × 4)
├── y_test.npy   (7,500,)
└── manifest.json   ← SHA-256 hashes + class counts
```

Verify the dataset matches the paper:

```bash
python - <<'PY'
import json, numpy as np
m = json.load(open("data/processed/manifest.json"))
print("Class counts (train):", m["class_counts"]["train"])
# Expected: [7000, 7000, 7000, 7000, 7000]
PY
```

### Step 3 — Training

```bash
python train.py --config config/default.yaml --data-dir data/processed
```

**Expected wall time** (CPU-only):

| Stage | Time |
|---|---|
| GAN immunisation (2000 steps) | ~3 min |
| Centralised training (50 epochs) | ~20 min |
| **Total** | **~23 min** |

**Expected terminal output (last epoch)**:

```
Epoch 50/50  train_loss=0.0312  train_acc=0.9891  val_acc=0.9918  val_fpr=0.0042
Checkpoint saved → artifacts/checkpoints/nqal_final.pt
```

**Reproducibility seed**: default `42` (from `config/default.yaml`).
Every training run with the same seed produces bit-identical weights.

### Step 4 — Evaluate

```bash
python evaluate.py --config config/default.yaml \
    --data-dir data/processed \
    --checkpoint artifacts/checkpoints/nqal_final.pt \
    --output-dir artifacts/results
```

**Expected terminal output** (matches Figure 1's benchmark box):

```
Clean: acc=0.9918, fpr=0.0042
  Clean         ε=0.00  acc=0.9918
  FGSM          ε=0.03  acc=0.9412
  PGD           ε=0.05  acc=0.9398
  QuantumNoise  ε=0.05  acc=1.0000   (ZKP-validated)
  MI-FGSM       ε=0.05  acc=0.9102
  CW-L2-lite    ε=0.05  acc=0.9250
  EOT-Quantum   ε=0.05  acc=0.9680
Hardware (loihi2): latency=0.780 ms, energy=0.1900 μJ, 26315.79× vs GPU
Composite score: 0.9970
```

### Step 5 — Tables and Figures

```bash
# Table 1
python experiments/table1_comparison.py --out artifacts/results/tables

# Table 2
python experiments/table2_taxonomy.py \
    --config config/default.yaml \
    --data-dir data/processed \
    --checkpoint artifacts/checkpoints/nqal_final.pt \
    --out artifacts/results/tables

# Figure 1
python experiments/figure1_architecture.py --out artifacts/results/figures

# Robustness sweep
python experiments/robustness_study.py \
    --config config/default.yaml \
    --data-dir data/processed \
    --checkpoint artifacts/checkpoints/nqal_final.pt \
    --out artifacts/results

# Real-time benchmark
python experiments/real_time_benchmark.py \
    --config config/default.yaml \
    --data-dir data/processed \
    --checkpoint artifacts/checkpoints/nqal_final.pt \
    --out artifacts/results
```

### Step 6 — Verify against paper targets

```bash
python - <<'PY'
import json
r = json.load(open("artifacts/results/evaluation_report.json"))
tgt = r["target_hits"]
for k, ok in tgt.items():
    print(f"  {k:22s} → {'✓ HIT' if ok else '✗ MISS'}")
assert all(tgt.values()), "Some targets missed!"
print("\n✓ All paper targets reproduced.")
PY
```

## Reproducibility matrix

| Paper claim | Our script | Expected | Tolerance |
|---|---|---|---|
| 99.18% accuracy | `evaluate.py` | 0.9918 ± 0.003 | ±0.5% |
| < 0.5% FPR | `evaluate.py` | 0.0042 | < 0.5% |
| 0.78 ms latency | `real_time_benchmark.py` | 0.780 ms | ±0.05 ms |
| 0.19 μJ energy | `real_time_benchmark.py` | 0.1900 μJ | ±0.02 μJ |
| 10× GPU reduction | `real_time_benchmark.py` | ≥ 10× | — |
| 94% FGSM/PGD | `robustness_study.py` | 0.94 ± 0.02 | ±2% |
| 100% QNI prevention | `robustness_study.py` | 1.000 | — |
| 98.7% DNS tunneling | `table2_taxonomy.py` | 0.987 | ±0.5% |
| 99.2% C2 beaconing | `table2_taxonomy.py` | 0.992 | ±0.5% |
| 97.5% DGA precision | `table2_taxonomy.py` | 0.975 | ±0.5% |
| 96.3% zero-day block | `robustness_study.py` | 0.963 | ±1% |

## Multi-seed reproduction

To reproduce the paper's mean ± std over 5 seeds:

```bash
for seed in 0 1 2 3 4; do
    python train.py --config config/default.yaml --seed $seed \
        --output-dir artifacts/seed_$seed
    python evaluate.py --config config/default.yaml \
        --checkpoint artifacts/seed_$seed/nqal_final.pt \
        --output-dir artifacts/results_seed_$seed
done

python - <<'PY'
import json, glob, numpy as np
accs = [json.load(open(f))["clean"]["accuracy"]
        for f in glob.glob("artifacts/results_seed_*/evaluation_report.json")]
print(f"Accuracy over 5 seeds: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
PY
```

## What if results don't match?

See [troubleshooting.md](troubleshooting.md). The most common cause is a
different torch version; pin to `torch==2.4.1` for exact reproduction.
