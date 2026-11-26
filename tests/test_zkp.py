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
    proof
