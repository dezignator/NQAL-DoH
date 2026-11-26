# Language & Technology Breakdown

The NQAL-DoH framework is inherently multi-disciplinary, requiring low-level hardware simulation, high-performance networking, cryptographic proving, and machine learning.

## Why Multiple Languages?

| Language | Component | Rationale |
|----------|-----------|-----------|
| **Python** | ML Core, GANs, QNI, Datasets | Standard for PyTorch and rapid research iteration. |
| **C++** | `cpp/lif_neuron_loihi.cpp` | Direct interaction with Intel Loihi 2 (Lava SDK) requires C++ for cycle-accurate simulation. |
| **C++** | `cpp/doh_packet_sim.cpp` | High-throughput DoH traffic generation requires libpcap and raw sockets. |
| **Rust** | `rust/zkp_verifier.rs` | Memory-safe, blazing-fast Groth16 pairing verification (Eq. 6) for zero-knowledge audits. |
| **Go** | `go/fl_coordinator.go` | High-concurrency Federated Learning coordinator (Eq. 7) handling 10,000+ concurrent client nodes. |
| **Circom** | `circuits/nqal_circuit.circom` | The actual zk-SNARK circuit (Eq. 5) compiled to R1CS for production. |
| **Docker/Bash** | `deploy/`, `scripts/` | Orchestration, CI/CD, and multi-node deployment. |

## Build Instructions
To build the entire multi-language stack:
```bash
chmod +x scripts/build_all.sh
./scripts/build_all.sh

---


