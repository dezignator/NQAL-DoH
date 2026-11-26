#!/usr/bin/env bash
# =============================================================================
# scripts/build_all.sh
# Builds the entire NQAL-DoH multi-language ecosystem.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "============================================================"
echo " Building NQAL-DoH Multi-Language Stack"
echo "============================================================"

# 1. Python Environment
echo "[1/5] Setting up Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. C++ Components
echo "[2/5] Compiling C++ Neuromorphic & Packet Simulators..."
if command -v g++ &> /dev/null; then
    g++ -O3 -march=native cpp/lif_neuron_loihi.cpp -o bin/lif_sim || echo "Warning: C++ build failed"
else
    echo "g++ not found, skipping C++ build."
fi

# 3. Rust ZKP Verifier
echo "[3/5] Compiling Rust ZKP Verifier..."
if command -v cargo &> /dev/null; then
    cd rust && cargo build --release && cd ..
    cp rust/target/release/zkp_verifier bin/ || true
else
    echo "cargo not found, skipping Rust build."
fi

# 4. Go FL Coordinator
echo "[4/5] Compiling Go Federated Learning Coordinator..."
if command -v go &> /dev/null; then
    cd go && go build -o ../bin/fl_coordinator fl_coordinator.go && cd ..
else
    echo "go not found, skipping Go build."
fi

# 5. Circom ZKP Circuit
echo "[5/5] Compiling Circom ZKP Circuit..."
if command -v circom &> /dev/null; then
    mkdir -p circuits/build
    circom circuits/nqal_circuit.circom --r1cs --wasm --sym -o circuits/build/
else
    echo "circom not found, skipping circuit build. (Install via npm install -g circom)"
fi

echo "============================================================"
echo " Build Complete! Binaries available in ./bin/"
ls -la bin/ 2>/dev/null || true
echo "============================================================"
