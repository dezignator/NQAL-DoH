#!/bin/bash

# NQAL-DoH Full Experiment Pipeline

set -e

echo "=== Step 1: Generating synthetic DoH dataset ==="
python data/generate_dataset.py

echo "=== Step 2: Training NQAL model (federated + adversarial) ==="
python train.py

echo "=== Step 3: Evaluating NQAL model ==="
python evaluate.py

echo "=== Experiment complete. Results saved. ==="
