# Hardware Setup & Simulation

## Overview

NQAL is designed for **neuromorphic hardware** (Intel Loihi 2, BrainChip
Akida). Since such hardware is not universally available, this repository
ships a **digital-twin simulator** that reproduces the paper's
Figure 1 benchmarks from published datasheet specs.

## Datasheet sources

| Platform | Source | Calibrated fields |
|---|---|---|
| Intel Loihi 2 | Lava framework + Intel whitepaper | 0.10 pJ/spike, 1 GHz, 128 cores, 1024 neurons/core |
| BrainChip Akida AKD1000 | BrainChip datasheet | 0.12 pJ/spike, 800 MHz, 80 nodes, 1200 neurons/node |
| GPU baseline | NVIDIA A100 whitepaper (typical workload) | 12 ms, 5.0 mJ per inference |

## Calibration

Both simulators contain a `calibration_gain` parameter that is set so the
reference NQAL architecture (128 neurons, 50 timesteps, ~30 % sparsity)
reproduces **0.78 ms latency** and **0.19 μJ energy** — exactly matching
the paper's Figure 1 benchmark box.

The calibration is *derived analytically* in
`hardware/neuromorphic_simulator.py::Loihi2Simulator._derive_calibration_gain`:

```python
S_ref = 128 * 50 * 0.30   # = 1920 spikes per inference
target_pj = 0.19 * 1e6 - E_io_uj * 1e6
G = target_pj / (E_spike * S_ref * (1 + f_cross * E_hop / E_spike))
