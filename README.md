# NQAL-DoH: Neuromorphic Quantum Adversarial Learning for DNS over HTTPS Threat Detection

This repository contains the official simulation implementation of the paper:

> **Neuromorphic Quantum Adversarial Learning (NQAL): a bio-inspired paradigm for DNS over HTTPS threat detection**  
> Basharat Ali, Guihai Chen  
> *EURASIP Journal on Information Security* (2025) 2025:35

## Overview

NQAL is a bio-inspired, zero-knowledge-supported detection mechanism that combines:
- Spiking Neural Networks (SNNs) with Dynamic Spiking Graph Attention (DSGAT)
- Quantum Noise Injection (QNI) for adversarial robustness
- Zero-Knowledge Proof Learning (ZKP-Learn) for privacy-preserving verification
- Swarm-Based Federated Learning for decentralized training
- Threat Immunization Engine using adversarial GANs

The framework detects DNS over HTTPS (DoH) threats including DNS tunneling, C2 beaconing, DGA, adversarial DoH, and quantum-noise injection exploits.

## Repository Structure

- `data/generate_dataset.py` – Generates synthetic DoH traffic with benign and malicious patterns.
- `models/` – Core NQAL components: SNN, DSGAT, QNI, ZKP, Federated Learning, GAN.
- `attacks/` – Adversarial attack implementations: FGSM, PGD, Quantum Noise.
- `utils/` – Metrics and energy estimation.
- `train.py` – Trains the NQAL model with federated learning and adversarial immunization.
- `evaluate.py` – Evaluates accuracy, FPR, robustness, latency, and energy.
- `run_experiments.sh` – Runs the complete pipeline.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NQAL-DoH.git
   cd NQAL-DoH
