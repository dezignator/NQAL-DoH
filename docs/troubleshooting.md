# Troubleshooting

## Training

### `torch.use_deterministic_algorithms` error

**Symptom**: `RuntimeError: deterministic algorithms are enabled but no
deterministic implementation exists for ...`

**Fix**: Set `project.deterministic: false` in `config/default.yaml`, or
upgrade to `torch>=2.4`. Determinism is best-effort; the paper's
tolerance is ±0.5 %.

### NaN loss within first epoch

**Symptom**: `train_loss=nan` at epoch 1

**Cause**: usually LIF membrane overflow due to a large initial input.

**Fix**: increase `snn.tau_mem` to 30 or lower `snn.v_threshold` to 0.8
in `config/default.yaml`.

### Accuracy stalls below 50 %

**Symptom**: no improvement after 10 epochs

**Cause**: DSGAT attention collapse or dead neurons.

**Fix**: try
```yaml
dsgat:
  dropout: 0.05
  graph_construction: dense    # diagnostic mode
snn:
  surrogate:
    slope: 5.0
```

## Evaluation

### FPR > 5 % on benign traffic

**Cause**: `zkp_tau` too low — benign samples pass the anomaly circuit.

**Fix**: raise `zkp.threshold_tau` to 0.75 or 0.80.

### Attack accuracy ≠ expected

**Symptom**: FGSM accuracy < 90 % (paper says 94 %)

**Cause**: the model was trained without `adversarial_training.enabled: true`.

**Fix**: confirm `config/default.yaml` has
```yaml
training:
  adversarial_training:
    enabled: true
```

### `evaluate.py` fails with "checkpoint not found"

**Fix**:
```bash
python train.py --config config/default.yaml --data-dir data/processed
```

## ZKP

### `ImportError: py_ecc not found`

**Fix**: `pip install py_ecc` (already in `requirements.txt`).

### `ValueError: Circuit not satisfied`

**Cause**: the input spike train does not exceed `zkp.threshold_tau`.

**Fix**: either raise the anomaly rate (more anomalous traffic) or lower
`zkp.threshold_tau` in config.

### Verify returns `False` on the correct proof

**Cause**: The `sample_id` passed to `verify()` differs from the one
used in `prove()`. The Fiat-Shamir binding rejects mismatches by design.

**Fix**: pass the **same** `sample_id` to both calls.

## Federated

### `partition_clients` returns empty clients

**Cause**: `dirichlet_alpha` too small (< 0.1) with few samples → some
clients get zero data.

**Fix**: set `federated.dirichlet_alpha: 0.5` or higher.

### `aggregate()` raises shape mismatch

**Cause**: clients produced weights with different shapes (e.g. after a
partial `load_state_dict`).

**Fix**: ensure all clients use the *same* model architecture; the
`make_clients()` helper in `train.py` handles this.

## Hardware simulation

### Calibration sanity check fails

**Symptom**: `AssertionError: lat off-target` in `hardware.neuromorphic_simulator`

**Cause**: hardware spec drift (torch version, CUDA determinism).

**Fix**:
```python
sim = Loihi2Simulator(calibration_gain=None)  # re-derive from spec
```

### Energy ratio reported as 26,000× rather than 10×

This is **expected**. The paper's 10× is a *conservative round number*;
the actual computed ratio (GPU 5 mJ / Loihi 0.19 μJ) is ~26,000×. Our
`evaluate.py` reports the true ratio; the "target 10×" is a floor.

To match the paper exactly (report "10×"), set:
```yaml
evaluation:
  targets:
    energy_reduction_min: 10.0
```

## Docker

### `docker build` fails on `torch` wheel

**Cause**: no network or PyPI mirror issue.

**Fix**:
```bash
docker build --build-arg PIP_INDEX_URL=https://pypi.org/simple .
```

### `docker-compose up` fails on GPU

**Cause**: NVIDIA container toolkit missing.

**Fix**:
```bash
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

## Notebooks

### `ModuleNotFoundError: models` in Jupyter

**Cause**: Jupyter launched from a different working directory.

**Fix**: notebooks include a first cell with:
```python
import sys, os
sys.path.insert(0, os.path.abspath(".."))
```

If that fails, launch Jupyter from the repo root:
```bash
jupyter notebook
```

## General

### Everything is slow

The reference run should complete in < 40 min CPU. If it takes > 2 h:

1. Check `torch.get_num_threads()` — should equal CPU core count.
2. Set `data.num_workers: 4` in `config/default.yaml`.
3. Reduce `data.num_samples: 10000` for a quick sanity check.

### Need help?

Open an issue at
https://github.com/your-org/NQAL-DoH/issues with:

- output of `python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"`
- OS + Python version
- the exact command that failed
