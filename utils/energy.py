"""
Energy consumption estimation.
Implements Eq. (9) from the paper.
"""

import torch

def compute_energy(spike_counts, weights, energy_per_spike_pj=0.1, sparsity_penalty_lambda=0.01):
    """
    Eq. (9): E_inf = sum(n_i * E_spike) + lambda * ||W||_0
    """
    snn_cost = sum(spike_counts) * energy_per_spike_pj
    sparsity_penalty = sparsity_penalty_lambda * torch.norm(weights, p=0).item()
    total_energy = snn_cost + sparsity_penalty
    return total_energy
