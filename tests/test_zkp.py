#!/usr/bin/env python3
"""tests/test_zkp.py — validates Eq. 5, 6, and privacy properties."""
import torch
import pytest

from models.zkp import ZKPLearn, ZKProof, Groth16Simulator
from py_ecc.bn128 import G1, G2, multiply, curve_order


def test_circuit_satisfied():
    zkp = ZKPLearn(threshold_tau=0.30, use_rate=True)
    x_hi = (torch.rand(50, 4) < 0.6).float()
    x_lo = (torch.rand(50, 4) < 0.05).float()
    assert zkp.circuit(x_hi) == 1
    assert zkp.circuit(x_lo) == 0


def test_prove_verify():
    zkp = ZKPLearn(threshold_tau=0.30)
    x = (torch.rand(50, 4) < 0.6).float()
    proof = zkp.prove(x, sample_id=7)
    assert zkp.verify(proof, sample_id=7)


def test_wrong_sample_id_rejected():
    zkp = ZKPLearn(threshold_tau=0.30)
    x = (torch.rand(50, 4) < 0.6).float()
    proof = zkp.prove(x, sample_id=7)
    assert not zkp.verify(proof, sample_id=8)


def test_tampered_proof_rejected():
    zkp = ZKPLearn(threshold_tau=0.30)
    x = (torch.rand(50, 4) < 0.6).float()
    proof = zkp.prove(x, sample_id=7)
    bad = ZKProof(
        pi_a=proof.pi_a,
        pi_b=proof.pi_b,
        pi_c=multiply(proof.pi_c, 2),
        public_input_hash=proof.public_input_hash,
        sample_id=proof.sample_id,
    )
    assert not zkp.verify(bad, sample_id=7)


def test_prove_unsatisfied_raises():
    zkp = ZKPLearn(threshold_tau=0.30)
    x = (torch.rand(50, 4) < 0.05).float()
    with pytest.raises(ValueError):
        zkp.prove(x, sample_id=0)


def test_privacy_audit():
    zkp = ZKPLearn(threshold_tau=0.30)
    x = (torch.rand(50, 4) < 0.6).float()
    audit = zkp.privacy_audit(x, sample_id=0)
    assert audit["verified"]
    assert audit["raw_data_exposed"] is False
    assert audit["spike_train_exposed"] is False
    assert audit["proof_size_bytes"] > 0


def test_groth16_pairing_equation():
    """Direct check that Eq. 6 holds for random (α, β, w, r)."""
    sim = Groth16Simulator(alpha_s=12345, beta_s=67890)
    for w, r in [(7, 3), (123456, 789), (curve_order // 2, 42)]:
        pi_a, pi_b, pi_c = sim.create_proof(w=w, r=r)
        assert sim.verify_pairing(pi_a, pi_b, pi_c), f"eq 6 failed for w={w}"


def test_proof_deterministic():
    """Same sample_id + same spike train → identical proof (reproducible)."""
    zkp = ZKPLearn(threshold_tau=0.30, seed=0)
    x = (torch.rand(50, 4) < 0.6).float()
    p1 = zkp.prove(x, sample_id=1)
    p2 = zkp.prove(x, sample_id=1)
    assert p1.pi_a == p2.pi_a
    assert p1.pi_b == p2.pi_b
    assert p1.pi_c == p2.pi_c
