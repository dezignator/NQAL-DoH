#!/usr/bin/env python3
"""
tests/test_lif_neuron.py
========================
Validates LIFNeuron membrane dynamics and STDPLearner convergence
(Theorem 2, Eq. 11).
"""
import math
import torch
import pytest

from models.lif_neuron import LIFNeuron, STDPLearner, SurrogateSpike


def test_membrane_decay():
    """Membrane potential must decay exponentially when input is zero."""
    lif = LIFNeuron(tau_mem=20.0)
    state = lif.init_state(1, 1, torch.device("cpu"))
    state.mem[0, 0] = 1.0
    # Apply zero current for 20 ticks → expect ≈ e⁻¹
    for _ in range(20):
        lif.step(torch.zeros(1, 1), state)
    assert 0.30 < state.mem.item() < 0.42


def test_refractory():
    """Neuron cannot fire while refractory counter > 0."""
    lif = LIFNeuron(refr_ticks=3, v_th=0.5)
    state = lif.init_state(1, 1, torch.device("cpu"))
    strong = torch.ones(1, 1) * 5.0
    s1 = lif.step(strong, state)
    s2 = lif.step(strong, state)
    s3 = lif.step(strong, state)
    s4 = lif.step(strong, state)
    s5 = lif.step(strong, state)
    # First spike then 3 refractory ticks → s1=1, s2..s4=0, s5=1
    assert s1.item() == 1.0
    for s in (s2, s3, s4):
        assert s.item() == 0.0
    assert s5.item() == 1.0


def test_surrogate_gradient_nonzero():
    v = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    s = SurrogateSpike.apply(v, 1.0, 10.0)
    s.sum().backward()
    assert v.grad is not None
    assert (v.grad.abs() > 0).any()


def test_stdp_sign():
    """Pre→post (t_pre < t_post) should strengthen the weight (LTP)."""
    stdp = STDPLearner(a_plus=0.1, a_minus=0.1, tau_plus=20.0, tau_minus=20.0)
    # 1 pre-spike at t=0, 1 post-spike at t=5 → LTP
    pre = torch.zeros(1, 10, 1); pre[0, 0, 0] = 1.0
    post = torch.zeros(1, 10, 1); post[0, 5, 0] = 1.0
    W = torch.zeros(1, 1)
    dw = stdp.trace_update(pre, post, W)
    assert dw.item() > 0


def test_stdp_anti_causal():
    """Post→pre (t_post < t_pre) should weaken the weight (LTD)."""
    stdp = STDPLearner(a_plus=0.1, a_minus=0.1, tau_plus=20.0, tau_minus=20.0)
    pre = torch.zeros(1, 10, 1); pre[0, 5, 0] = 1.0
    post = torch.zeros(1, 10, 1); post[0, 0, 0] = 1.0
    W = torch.zeros(1, 1)
    dw = stdp.trace_update(pre, post, W)
    assert dw.item() < 0


def test_theorem2_convergence():
    """Effective threshold must converge (std of late window < early window)."""
    stdp = STDPLearner()
    W = torch.rand(32, 32) * 0.5
    pre = (torch.rand(8, 50, 32) < 0.2).float()
    post = (torch.rand(8, 50, 32) < 0.2).float()
    eff = []
    for step in range(40):
        dw = stdp.trace_update(pre, post, W)
        stdp.apply(W, dw)
        diag = stdp.convergence_diagnostics(W, step)
        eff.append(diag["eff_threshold"])
    late_std = torch.tensor(eff[-10:]).std().item()
    early_std = torch.tensor(eff[:10]).std().item()
    assert late_std <= early_std + 1e-6
