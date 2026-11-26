"""
Quantum Noise Injection (QNI) attack.
Simulates quantum-style perturbations.
"""

import torch
import numpy as np

def quantum_noise_attack(x, epsilon=0.05):
    """Adds quantum random walk noise to input."""
    # Simulate quantum random walk noise
    noise = torch.randn_like(x) * epsilon
    # Apply Hadamard-like transformation (simplified)
    noise = torch.sign(noise) * torch.abs(noise) ** 0.5
    return torch.clamp(x + noise, 0, 1)
