# NQAL-DoH: Neuromorphic Quantum Adversarial Learning for DNS over HTTPS Threat Detection

<p align="center">
  <img src="docs/architecture.png" alt="NQAL Architecture" width="900"/>
</p>

**Official simulation repository for the paper:**

> **Neuromorphic Quantum Adversarial Learning (NQAL): a bio-inspired paradigm for DNS over HTTPS threat detection**  
> Basharat Ali, Guihai Chen  
> *EURASIP Journal on Information Security* (2025) 2025:35  
> DOI: 10.1186/s13635-025-00215-5

---

## 1. What This Repository Reproduces

This repository is a **full, executable re-implementation** of the NQAL framework described in the paper. It reproduces:

| Paper artifact | Reproduced in | Status |
|---|---|---|
| Eq. (1) – Neuromorphic spike train encoding | `models/encoding.py` | ✅ |
| Eq. (2) – Dynamic Spiking Graph Attention (DSGAT) | `models/dsgat.py` | ✅ |
| Eq. (3) – Quantum Random Walk noise injection | `models/qni.py` | ✅ |
| Eq. (4) – Adversarial loss $\mathcal{L}_{QAI}$ | `models/qni.py` | ✅ |
| Eq. (5) – zk-SNARK circuit $C(x,w)$ | `models/zkp.py` | ✅ |
| Eq. (6) – Groth16 pairing verification | `models/zkp.py` | ✅ |
| Eq. (7) – Swarm federated update rule | `models/federated.py` | ✅ |
| Eq. (8) – Adversarial GAN objective | `models/gan.py` | ✅ |
| Eq. (9) – Energy efficiency $E_{inf}$ | `utils/energy.py` | ✅ |
| Eq. (10) – QNI robustness bound | `experiments/theorems.py` | ✅ |
| Eq. (11) – STDP convergence | `models/snn.py` | ✅ |
| Table 1 – Comparative analysis | `experiments/table1_comparison.py` | ✅ |
| Table 2 – DoH attack taxonomy & mitigation | `experiments/table2_taxonomy.py` | ✅ |
| Figure 1 – Logical flow / architecture | `experiments/figure1_architecture.py` | ✅ |
| 99.18% accuracy, <1 ms latency | `evaluate.py` | ✅ |
| 10× energy vs GPU | `utils/energy.py` | ✅ |
| 94% FGSM/PGD robustness | `attacks/orchestrator.py` | ✅ |
| 100% QNI exploit prevention | `models/zkp.py` | ✅ |

---

## 2. Repository Layout
