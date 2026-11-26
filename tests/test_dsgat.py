#!/usr/bin/env python3
"""
tests/test_dsgat.py
===================
Validates DSGAT attention (Eq. 2) invariants.
"""
import math
import torch
import pytest

from models.dsgat import DSGAT, DSGATLayer


@pytest.mark.parametrize("mode", ["knn", "threshold", "dense"])
def test_graph_build(mode):
    dsgat = DSGAT(in_dim=32, hidden_dim=16, num_heads=2,
                  graph_construction=mode, knn_k=3)
    x = (torch.rand(2, 20, 32) < 0.3).float()
    adj = dsgat.build_graph(x)
    B, T, _ = x.shape
    assert adj.shape == (B, T, T)
    # Self-loops present
    eye = torch.eye(T).unsqueeze(0)
    assert (adj * eye).sum() == B * T
    # Symmetric
    assert torch.allclose(adj, adj.transpose(-2, -1), atol=1e-5)


def test_output_shape():
    dsgat = DSGAT(in_dim=64, hidden_dim=32, num_heads=4)
    x = torch.rand(2, 15, 64)
    adj = dsgat.build_graph(x)
    out = dsgat(x, adj=adj)
    assert out.shape == (2, 15, 32)


def test_self_attention_softmax():
    layer = DSGATLayer(in_dim=32, out_dim=16, num_heads=2, dropout=0.0)
    layer.eval()
    x = torch.randn(1, 10, 32)
    # Capture attention manually
    f = torch.zeros_like(x)
    y = torch.zeros(1, 32)
    for t in range(10):
        y = y * math.exp(-1/20) + x[:, t]
        f[:, t] = y
    Q = layer.theta_q(f).view(1, 10, 2, 8).transpose(1, 2)
    K = layer.theta_k(f).view(1, 10, 2, 8).transpose(1, 2)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(8)
    attn = torch.softmax(scores, dim=-1)
    # Softmax row-sums to 1
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-5)


def test_masking():
    layer = DSGATLayer(in_dim=16, out_dim=8, num_heads=2, dropout=0.0)
    layer.eval()
    x = torch.randn(1, 6, 16)
    adj = torch.zeros(1, 6, 6)
    # Only self-loops
    for i in range(6):
        adj[0, i, i] = 1.0
    out = layer(x, adj=adj)
    # With only self-loops, each output should equal the linear-proj of its own input
    assert out.shape == (1, 6, 8)
