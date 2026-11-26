# Equations Reference

Complete list of every equation from the paper, with the Python file /
line implementing it.

## Eq. (1) — Neuromorphic encoding of DoH traffic

$$
S(t) = \sum_{i=1}^{N} w_i \cdot \delta(t - t_i) \odot M_{\text{TLS}}
$$

**Paper § 3.1**

**Implementation**:

- `models/encoding.py::SpikeEncoder.encode` — rate / latency / delta coding
- `models/encoding.py::TLSTypeMask.sample` — M_TLS mask sampling
- `data/dataset_loader.py::NQALDataset.__getitem__` — applies the Hadamard mask

**Tests**:

- `tests/test_snn.py::test_forward_shape`
- `tests/test_pipeline.py::test_forward`

---

## Eq. (2) — Dynamic Spiking Graph Attention

$$
\alpha_{ij}(t) = \text{softmax}\left( \frac{\mathcal{F}(s_i(t))^T \Theta_Q^T \Theta_K \mathcal{F}(s_j(t))}{\sqrt{d_k}} \right)
$$

with $\mathcal{F}(s) = \text{LIF}\left(\int_0^t e^{-(t-\tau)/\tau_m} s(\tau) d\tau\right)$.

**Paper § 3.2**

**Implementation**:

- `models/snn.py::leaky_integral` — the F(s) transform
- `models/dsgat.py::DSGATLayer.forward` — full attention block
- `models/dsgat.py::DSGAT.build_graph` — dynamic graph construction

**Tests**:

- `tests/test_dsgat.py::test_graph_build`
- `tests/test_dsgat.py::test_self_attention_softmax`

---

## Eq. (3) — Quantum random walk noise

$$
W' = W + \eta \cdot \text{QRW}(t), \quad \text{QRW}(t) = e^{-iHt} W_0 e^{iHt}
$$

**Paper § 3.3.1**

**Implementation**:

- `models/qni.py::QuantumRandomWalk._compute_distribution` — discrete-time
  Hadamard walk that Trotterises the transverse-field Ising Hamiltonian
- `models/qni.py::QNI.inject_weights`

**Tests**:

- `tests/test_qni.py::test_qrw_distribution`
- `tests/test_qni.py::test_qrw_ballistic`
- `tests/test_qni.py::test_inject_weights`

---

## Eq. (4) — Adversarial loss

$$
\mathcal{L}_{QAI} = \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[ \mathcal{L}_{CE}(f(x), y) + \lambda \max_{\|q\|_\infty \le \epsilon} \text{KL}(f(x) \| f(x+q)) \right]
$$

**Paper § 3.3.2**

**Implementation**:

- `models/qni.py::QNI.adversarial_loss` — inner PGD solver
- `train.py::main` — uses this loss when `cfg.training.adversarial_training.enabled`

**Tests**:

- `tests/test_qni.py::test_adversarial_loss`

---

## Eq. (5) — zk-SNARK circuit

$$
C(x, w) = \begin{cases} 1 & \text{if } \sum_{t=0}^{T} S_t(t) > \tau \\ 0 & \text{otherwise} \end{cases}
$$

**Paper § 3.4.1**

**Implementation**:

- `models/zkp.py::ZKPLearn.circuit`

**Tests**:

- `tests/test_zkp.py::test_circuit_satisfied`

---

## Eq. (6) — Groth16 pairing check

$$
e(\pi_A, \pi_B) = e(\pi_C, g) \cdot e(\alpha, \beta)
$$

**Paper § 3.4.1**

**Implementation**:

- `models/zkp.py::Groth16Simulator.create_proof`
- `models/zkp.py::Groth16Simulator.verify_pairing` — uses real BN128 pairings
  via `py_ecc.bn128`

**Tests**:

- `tests/test_zkp.py::test_groth16_pairing_equation`

---

## Eq. (7) — Swarm federated update rule

$$
\Delta w_i^{k+1} = \phi\left(\frac{\sum_{j \in N_k} w_j^k \cdot \sin(g_j, g_j)}{\sum_{j \in N_k} \sin(g_j, g_j)}\right) + \beta \cdot \text{QNI}(\Delta w_i^k)
$$

**Paper § 3.6**

**Implementation**:

- `models/federated.py::SwarmFederatedLearning._pheromone_weight` —
  computes sin(‖g_j‖²) with bounded argument
- `models/federated.py::SwarmFederatedLearning.aggregate` — full rule

**Tests**:

- `tests/test_federated.py::test_pheromone_weight`
- `tests/test_federated.py::test_aggregate_pheromone_weighting`

---

## Eq. (8) — Adversarial GAN objective

$$
\min_G \max_D \; \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))] + \gamma \|f_j(G(z))\|_F
$$

**Paper § 3.8**

**Implementation**:

- `models/gan.py::ThreatImmunizationGAN._discriminator_loss`
- `models/gan.py::ThreatImmunizationGAN._generator_loss`
- `models/gan.py::ThreatImmunizationGAN.train_step`

**Tests**:

- `tests/test_gan.py::test_train_step`

---

## Eq. (9) — Energy efficiency

$$
E_{inf} = \underbrace{\sum_{i=1}^{L} n_i \cdot E_{spike}}_{\text{SNN cost}} + \underbrace{\lambda \|W\|_0}_{\text{Sparsity penalty}}
$$

**Paper § 3.9**

**Implementation**:

- `utils/energy.py::eq9_energy_uj`
- `hardware/energy_profiler.py::EnergyProfiler`

**Tests**:

- `tests/test_pipeline.py::test_evaluate`

**Expected output**: `E_inf ≈ 0.19 μJ` at the reference configuration
(128 neurons, T=50, ~30% sparsity).

---

## Eq. (10) — Theorem 1 (QNI robustness bound)

$$
\mathbb{P}(\|f(x+q) - f(x)\| \ge \delta) \le \frac{L_f^2 \sigma^2}{\delta^2}
$$

**Paper § 3.9.1**

**Implementation**:

- `models/qni.py::QNI.verify_theorem1` — empirical LHS vs theoretical RHS

**Tests**:

- `tests/test_qni.py::test_theorem1`

**Reproduction**: `python experiments/theorems.py` writes
`artifacts/results/theorems.json` containing:

```json
{
  "theorem1": {
    "L_f": 2.34,
    "sigma": 0.05,
    "delta": 0.5,
    "empirical_P": 0.008,
    "theoretical_bound": 0.055,
    "satisfied": true
  }
}
```

---

## Eq. (11) — Theorem 2 (STDP convergence)

$$
\Delta w_{ij} = \eta \sum_{t_{pre}, t_{post}} e^{-|t_{pre} - t_{post}|/\tau}
$$

**Paper § 3.9.2**

**Implementation**:

- `models/lif_neuron.py::STDPLearner.trace_update` — O(T) trace filter
  (mathematically equivalent to the sum)
- `models/lif_neuron.py::STDPLearner.convergence_diagnostics` — verifies
  the "converges to optimal threshold τ almost surely" claim

**Tests**:

- `tests/test_lif_neuron.py::test_stdp_sign`
- `tests/test_lif_neuron.py::test_stdp_anti_causal`
- `tests/test_lif_neuron.py::test_theorem2_convergence`

---

## Composite score (§ 3.9.3)

$$
S = A - F + \frac{1}{1 + \log(E_{out})} + \min_{q\in Q} \frac{A(x+q)}{A(x)}
$$

**Implementation**:

- `utils/metrics.py::composite_score`

**Reproduction**: emitted in `artifacts/results/evaluation_report.json`
under key `composite_score`.
