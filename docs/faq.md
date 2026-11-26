# FAQ

### Is this the *actual* paper code?

This is a **faithful independent re-implementation** based on the paper's
public description (EURASIP J. Inf. Secur., 2025:35). The authors did not
release a reference implementation at time of writing. Our reproduction
matches every reported metric within tolerance (see
`docs/reproduction_guide.md` for the tolerance matrix).

### Does it require real neuromorphic hardware?

No. `hardware/neuromorphic_simulator.py` provides digital-twin
simulators for Intel Loihi 2 and BrainChip Akida. For users with actual
devices, the simulator interface is designed to be drop-in replaceable.

### Is the ZKP cryptographically secure?

The pairing equation (Eq. 6) uses **real BN128 elliptic curve
operations** via `py_ecc`. However:

- The **trusted setup** is not real — we derive (α, β) from a SHA-256
  hash for reproducibility.
- The **prover is honest** by assumption (this matches the paper's
  threat model — the prover is the trusted NQAL model, the verifier is
  an external auditor).

For production, replace `Groth16Simulator.__init__` with a trusted-setup
output from `snarkjs` or `bellman`. The verification equation itself is
unchanged.

### Can I train on real DoH traffic?

Yes. Replace `data/generate_dataset.py` with a loader for
- CIRA-CIC-DoHBrw-2020
- dahl_doh
- doh_resolvers_2022

and ensure the output shape is `(N, 50, 4)` with channels
`(packet_size, iat_ms, entropy_bits, direction)`. Then `train.py` works
unchanged.

### Why 50 timesteps?

This matches the paper's § 4.1 setup. It corresponds to a 50-packet
window, roughly 2.5 s of DoH traffic at 20 pps. Longer windows help C2
detection (heartbeat period ~8 s) but require more state. 50 is a good
trade-off.

### Why does the energy ratio show 26,000× instead of 10×?

See `docs/hardware_setup.md`. The paper reports a conservative 10× floor;
the actual computed ratio is ~26,000×. `evaluate.py` reports both.

### How do I regenerate just one table?

```bash
# Table 1 only
python experiments/table1_comparison.py

# Table 2 only (needs trained checkpoint)
python experiments/table2_taxonomy.py
